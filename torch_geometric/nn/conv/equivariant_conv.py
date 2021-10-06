from typing import Optional, Callable, Tuple
from torch_geometric.typing import OptTensor, Adj, SparseTensor

import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import reset


def get_fully_connected_edges(n_nodes: int, add_self_loops: bool = False) -> Tensor:
    """
    Creates the edge_index in COO format in a fully-connected graph with :obj:`n_nodes` nodes.
    """
    edge_index = torch.cartesian_prod(torch.arange(n_nodes), torch.arange(n_nodes)).T
    if not add_self_loops:
        edge_index = edge_index.t()[edge_index[0] != edge_index[1]].t()

    return edge_index


def get_fully_connected_edges_in_batch(batch: Tensor,
                                       edge_index: Tensor,
                                       fill_value: float = 0.0,
                                       add_self_loops: bool = False,
                                       edge_attr: OptTensor = None) -> Tuple[Tensor, Tensor, OptTensor]:
    """
    Creates the edge_index in COO format in a fully-connected graph in a batch
    """

    # create number of graphs in batch
    # batch_size = len(batch.unique())
    # batch_num_nodes = torch.zeros(size=(batch_size,), device=batch.device, dtype=batch.dtype)
    # ones = torch.ones_like(batch)
    # batch_num_nodes = batch_num_nodes.scatter_add(0, batch, ones)

    batch_num_nodes = torch.bincount(batch)

    # create pointer (ptr)
    zero = torch.zeros(size=(1,), dtype=batch.dtype, device=batch.device)
    ptr = batch_num_nodes.cumsum(dim=0)
    ptr = torch.cat([zero, ptr])


    fc_edge_index = [get_fully_connected_edges(int(n), add_self_loops) + int(p) for n, p in zip(batch_num_nodes, ptr)]
    fc_edge_index = torch.cat(fc_edge_index, dim=-1)


    # create dictionary with string as keys, e.g. [0, 1] meaning the connectivity between source node_id 0 to 1
    # the value of the position along dim=1 of edge_index
    source_target_to_edge_idx = {str([int(s), int(t)]): i for s, t, i in zip(edge_index[0],
                                                                             edge_index[1],
                                                                             range(edge_index.size(1)))}
    # positions of fake edge_index
    source_target_to_fc_edge_idx = {str([int(s), int(t)]): i for s, t, i in zip(fc_edge_index[0],
                                                                                fc_edge_index[1],
                                                                                range(fc_edge_index.size(1)))}

    fake_edges = [s for s in source_target_to_fc_edge_idx.keys() if s not in source_target_to_edge_idx.keys()]
    fake_edges_ids = [source_target_to_fc_edge_idx[k] for k in fake_edges]

    E = fc_edge_index.size(-1)
    true_edge_ids = list(set([i for i in range(E)]).difference(set(fake_edges_ids)))
    # create mask where the true edge-indices are placed
    mask = torch.ones(size=(E,), device=batch.device, dtype=torch.bool)
    # fill in mask
    mask[fake_edges_ids] = False

    # print(fc_edge_index)
    # print(edge_index)
    # print(mask)

    if edge_attr is not None:
        # concatenate/zero-pad the fake edge_attr behind the true edge_attr
        all_edge_attr = torch.zeros(size=(fc_edge_index.size(1), edge_attr.size(-1)),
                                    device=edge_index.device).fill_(fill_value)
        # fill in values from the true edge attr
        all_edge_attr[true_edge_ids] = edge_attr
    else:
        all_edge_attr = None

    return fc_edge_index, mask, all_edge_attr


class EquivariantConv(MessagePassing):
    r"""The Equivariant graph neural network operator form the
    `"E(n) Equivariant Graph Neural Networks"
    <https://arxiv.org/pdf/2102.09844.pdf>`_ paper.

    .. math::
        \mathbf{m}_{ij}=h_{\mathbf{\Theta}}(\mathbf{x}_i,\mathbf{x}_j,\|
        {\mathbf{pos}_i-\mathbf{pos}_j}\|^2_2,\mathbf{e}_{ij})

        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}}(\mathbf{x}_i,
        \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij})

        \mathbf{vel}^{\prime}_i = \phi_{\mathbf{\Theta}}(\mathbf{x}_i)\mathbf
        {vel}_i + \frac{1}{|\mathcal{N}(i)|}\sum_{j \in\mathcal{N}(i)}
        (\mathbf{pos}_i-\mathbf{pos}_j)
        \rho_{\mathbf{\Theta}}(\mathbf{m}_{ij})

        \mathbf{pos}^{\prime}_i = \mathbf{pos}_i + \mathbf{vel}_i

    where :math:`\gamma_{\mathbf{\Theta}}`,
    :math:`h_{\mathbf{\Theta}}`, :math:`\rho_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote neural
    networks, *.i.e.* MLPs. :math:`\mathbf{P} \in \mathbb{R}^{N \times D}`
    and :math:`\mathbf{V} \in \mathbb{R}^{N \times D}`
    defines the position and velocity of each point respectively.

    Args:
        local_nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x`,
            sqaured distance :math:`\|{\mathbf{pos}_i-\mathbf{pos}_j}\|^2_2`
            and edge_features :obj:`edge_attr`
            of shape :obj:`[-1, 2*in_channels + 1 +edge_dim]`
            to shape :obj:`[-1, hidden_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        pos_nn (torch.nn.Module,optional): A neural network
            :math:`\rho_{\mathbf{\Theta}}` that
            maps message :obj:`m` of shape
            :obj:`[-1, hidden_channels]`,
            to shape :obj:`[-1, 1]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        vel_nn (torch.nn.Module,optional): A neural network
            :math:`\phi_{\mathbf{\Theta}}` that
            maps node featues :obj:`x` of shape :obj:`[-1, in_channels]`,
            to shape :obj:`[-1, 1]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): A neural network
            :math:`\gamma_{\mathbf{\Theta}}` that maps
            message :obj:`m` after aggregation
            and node features :obj:`x` of shape
            :obj:`[-1, hidden_channels + in_channels]`
            to shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        aggr (string, optional): The operator used to aggregate message for hidden node embeddings
            :obj:`m` (:obj:`"add"`, :obj:`"mean"`).
            (default: :obj:`"add"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, local_nn: Optional[Callable] = None,
                 pos_nn: Optional[Callable] = None,
                 vel_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 aggr: str = "add",
                 use_fc: bool = True,
                 **kwargs):
        super(EquivariantConv, self).__init__(aggr=aggr, **kwargs)

        self.local_nn = local_nn
        self.pos_nn = pos_nn
        self.vel_nn = vel_nn
        self.global_nn = global_nn

        self.use_fc = use_fc

        self.reset_parameters()

    def reset_parameters(self):

        reset(self.local_nn)
        reset(self.pos_nn)
        reset(self.vel_nn)
        reset(self.global_nn)

    def forward(self, x: OptTensor,
                pos: Tensor,
                edge_index: Adj,
                vel: OptTensor = None,
                batch: OptTensor = None,
                edge_attr: OptTensor = None
                ) -> Tuple[Tensor, Tuple[Tensor, OptTensor]]:
        """"""

        if isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
            edge_index = torch.stack([row, col], dim=0)

        if batch is None:
            batch = torch.zeros([pos.size(0)], device=pos.device, dtype=torch.long)

        # create fully-connected graph(s) in batch

        fc_edge_index, mask, edge_attr = get_fully_connected_edges_in_batch(batch=batch,
                                                                            edge_index=edge_index,
                                                                            add_self_loops=False,
                                                                            edge_attr=edge_attr)


        # propagate_type: (x: OptTensor, pos: Tensor, edge_attr: OptTensor) -> Tuple[Tensor,Tensor] # noqa
        out_x, out_pos = self.propagate(edge_index=fc_edge_index,
                                        orig_index=edge_index[1],
                                        edge_mask=mask,
                                        x=x, pos=pos,
                                        edge_attr=edge_attr, size=None)

        out_x = out_x if x is None else torch.cat([x, out_x], dim=1)
        if self.global_nn is not None:
            out_x = self.global_nn(out_x)

        if vel is None:
            out_pos += pos
            out_vel = None
        else:
            out_vel = (vel if self.vel_nn is None or x is None else
                       self.vel_nn(x) * vel) + out_pos
            out_pos = pos + out_vel

        return (out_x, (out_pos, out_vel))

    def message(self, x_i: OptTensor, x_j: OptTensor, pos_i: Tensor,
                pos_j: Tensor,
                edge_attr: OptTensor = None) -> Tuple[Tensor, Tensor]:

        msg = torch.sum((pos_i - pos_j).square(), dim=1, keepdim=True)
        msg = msg if x_j is None else torch.cat([x_j, msg], dim=1)
        msg = msg if x_i is None else torch.cat([x_i, msg], dim=1)
        msg = msg if edge_attr is None else torch.cat([msg, edge_attr], dim=1)
        msg = msg if self.local_nn is None else self.local_nn(msg)

        pos_msg = ((pos_i - pos_j) if self.pos_nn is None else
                   (pos_i - pos_j) * self.pos_nn(msg))
        return (msg, pos_msg)

    def aggregate(self, inputs: Tuple[Tensor, Tensor],
                  index: Tensor, edge_mask: Tensor) -> Tuple[Tensor, Tensor]:

        msgs_feat = inputs[0].masked_select(edge_mask.view(-1, 1)).view(-1, inputs[0].size(-1))
        msgs_pos = inputs[1]
        orig_index = index.masked_select(edge_mask)
        node_aggr_messages = scatter(msgs_feat, orig_index, 0, reduce=self.aggr)
        node_aggr_positional = scatter(msgs_pos, index, 0, reduce="mean")
        return (node_aggr_messages, node_aggr_positional)

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs

    def __repr__(self):
        return ("{}(local_nn={}, pos_nn={},"
                " vel_nn={},"
                " global_nn={})").format(self.__class__.__name__,
                                         self.local_nn, self.pos_nn,
                                         self.vel_nn, self.global_nn)
