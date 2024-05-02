import torch
from metrics.miss_rate import MissRate


def test_miss_rate_multimodal() -> None:
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))

    mr = MissRate()
    for msk in [None, mask]:
        mr.update(pred, trg, msk)
        mr.compute()


def test_miss_rate_unimodal() -> None:
    batch_size, seq_len, num_dims = 32, 25, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))

    mr = MissRate()
    for msk in [None, mask]:
        mr.update(pred, trg, msk)
        mr.compute()
