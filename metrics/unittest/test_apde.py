import torch
from metrics.min_apde import MinAPDE


def test_min_apde_multimodal() -> None:
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))

    min_apde = MinAPDE()
    for msk in [None, mask]:
        min_apde.update(pred, trg, msk)
        min_apde.compute()


def test_min_apde_unimodal() -> None:
    batch_size, seq_len, num_dims = 32, 25, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))

    min_apde = MinAPDE()
    for msk in [None, mask]:
        min_apde.update(pred, trg, msk)
        min_apde.compute()
