import torch
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_geometric.nn import EquivariantConv


def test_equivariant_conv():
    x1 = torch.randn(4, 16)
    pos1 = torch.randn(4, 3)
    vel = torch.randn(4, 3)
    # edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_index = torch.tensor([[1, 0, 0, 2, 1, 3], [0, 1, 2, 0, 3, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    pos_nn = Linear(32, 1)
    vel_nn = Linear(16, 1)
    local_nn = Linear(2 * 16 + 1, 32)
    global_nn = Linear(16 + 32, 16)

    conv = EquivariantConv(local_nn, pos_nn, vel_nn, global_nn)
    assert conv.__repr__() == (
        'EquivariantConv('
        'local_nn=Linear(in_features=33, out_features=32, bias=True),'
        ' pos_nn=Linear(in_features=32, out_features=1, bias=True),'
        ' vel_nn=Linear(in_features=16, out_features=1, bias=True),'
        ' global_nn=Linear(in_features=48, out_features=16, bias=True))')
    out = conv(x1, pos1, edge_index, vel)
    assert out[0].size() == (4, 16)
    assert out[1][0].size() == (4, 3)
    assert out[1][1].size() == (4, 3)
    assert torch.allclose(conv(x1, pos1, edge_index)[0], out[0], atol=1e-6)
    assert conv(x1, pos1, edge_index)[1][1] is None
    assert torch.allclose(conv(x1, pos1, adj.t(), vel)[0], out[0], atol=1e-6)
    assert torch.allclose(conv(x1, pos1, adj.t())[0], out[0], atol=1e-6)

    t = ('(Tensor, Tensor, Tensor, OptTensor, NoneType)'
         ' -> Tuple[Tensor, Tuple[Tensor, OptTensor]]')
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x1, pos1, edge_index, vel)[0], out[0], atol=1e-6)
    assert torch.allclose(jit(x1, pos1, edge_index, vel)[1][0], out[1][0],
                          atol=1e-6)
    assert torch.allclose(jit(x1, pos1, edge_index, vel)[1][1], out[1][1],
                          atol=1e-6)

    t = ('(Tensor, Tensor, Tensor, NoneType, NoneType)'
         ' -> Tuple[Tensor, Tuple[Tensor, OptTensor]]')
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x1, pos1, edge_index)[0], out[0], atol=1e-6)
    assert jit(x1, pos1, edge_index)[1][1] is None

    t = ('(Tensor, Tensor, SparseTensor, OptTensor, NoneType)'
         ' -> Tuple[Tensor, Tuple[Tensor, OptTensor]]')
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x1, pos1, adj.t(), vel)[0], out[0], atol=1e-6)
    assert torch.allclose(jit(x1, pos1, adj.t(), vel)[1][0], out[1][0],
                          atol=1e-6)
    assert torch.allclose(jit(x1, pos1, adj.t(), vel)[1][1], out[1][1],
                          atol=1e-6)


def test_equivariant_conv_fc():

    pos = torch.tensor([[0.0],
                        [1.0],
                        [2.0],
                        [3.0]
                        ])


    edge_index = torch.tensor([[1, 0, 0, 2, 1, 3], [0, 1, 2, 0, 3, 1]])
    conv = EquivariantConv(local_nn=None, pos_nn=None, vel_nn=None, global_nn=None)
    assert sum(m.numel() for m in conv.parameters()) == 0

    out = conv(x=None, pos=pos, edge_index=edge_index, vel=None, edge_attr=None)

    x_feat = out[0]
    pos_feat = out[1][0]

    def euclid_dist_unidim(x1: float, x2: float) -> float:
        return (x1-x2)**2


    # get fully-connected messages.
    # graph has N=4 nodes, so in-total (N*(N-1)/2) = 6 messages, as no self-loops are considered
    m01 = m10 = euclid_dist_unidim(pos[0].item(), pos[1].item())
    m02 = m20 = euclid_dist_unidim(pos[0].item(), pos[2].item())
    m03 = m30 = euclid_dist_unidim(pos[0].item(), pos[3].item())

    m12 = m21 = euclid_dist_unidim(pos[1].item(), pos[2].item())
    m13 = m31 = euclid_dist_unidim(pos[1].item(), pos[3].item())

    m23 = m32 = euclid_dist_unidim(pos[2].item(), pos[3].item())

    # get hidden embedding `h`, based on local neighbourhood.
    h0 = m02 + m01
    h1 = m01 + m13
    h2 = m02
    h3 = m13
    assert torch.allclose(x_feat, torch.Tensor([[h0], [h1], [h2], [h3]]))


    # get hidden positions `x`, based on the fully-connected graph.
    x = pos
    x0 = x[0].item() + (1/3) * sum([x[0].item() - (x[1].item()),
                                    x[0].item() - (x[2].item()),
                                    x[0].item() - (x[3].item())
                                    ])

    x1 = x[1].item() + (1/3) * sum([x[1].item() - (x[0].item()),
                                    x[1].item() - (x[2].item()),
                                    x[1].item() - (x[3].item())
                                    ])

    x2 = x[2].item() + (1/3) * sum([x[2].item() - (x[0].item()),
                                    x[2].item() - (x[1].item()),
                                    x[2].item() - (x[3].item())
                                    ])

    x3 = x[3].item() + (1 / 3) * sum([x[3].item() - (x[0].item()),
                                      x[3].item() - (x[1].item()),
                                      x[3].item() - (x[2].item())
                                      ])

    assert torch.allclose(pos_feat, torch.Tensor([[x0], [x1], [x2], [x3]]))
