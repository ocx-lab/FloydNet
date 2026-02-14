import torch


def test_readme_minimal_example() -> None:
    from floydnet import PivotalAttentionBlock, pivotal_attention

    b, n, c = 1, 4, 16
    x = torch.randn(b, n, n, c)
    module = PivotalAttentionBlock(embed_dim=c, num_heads=4, dropout=0.0)
    out = module(x)
    assert out.shape == (b, n, n, c)

    h, d = 4, 8
    q_ik = torch.randn(b, h, n, n, d)
    k_ij = torch.randn(b, h, n, n, d)
    k_jk = torch.randn(b, h, n, n, d)
    v_ij = torch.randn(b, h, n, n, d)
    v_jk = torch.randn(b, h, n, n, d)
    y = pivotal_attention(q_ik, k_ij, k_jk, v_ij, v_jk)
    assert y.shape == (b, h, n, n, d)
