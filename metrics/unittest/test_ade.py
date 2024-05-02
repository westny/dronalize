import torch
from metrics.min_ade import MinADE


def test_min_ade_multimodal() -> None:
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))

    min_ade = MinADE()
    for msk in [None, mask]:
        min_ade.update(pred, trg, msk)
        min_ade.compute()


def test_min_ade_unimodal() -> None:
    batch_size, seq_len, num_dims = 32, 25, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))

    min_ade = MinADE()
    for msk in [None, mask]:
        min_ade.update(pred, trg, msk)
        min_ade.compute()
