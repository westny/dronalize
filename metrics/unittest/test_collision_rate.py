import pytest
import torch
from metrics.collision_rate import CollisionRate


def test_collision_rate_multimodal() -> None:
    n_scenarios = 10
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)

    ptr = torch.randint(1, batch_size, (n_scenarios - 1,)).unique().sort()[0]
    ptr = torch.cat([torch.tensor([0]), ptr, torch.tensor([batch_size])])

    cr = CollisionRate()
    cr.update(pred, trg, ptr)
    cr.compute()


def test_collision_rate_unimodal() -> None:
    n_scenarios = 10
    batch_size, seq_len, num_dims = 32, 25, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    pred = torch.randn(batch_size, seq_len, num_dims)

    ptr = torch.randint(1, batch_size, (n_scenarios - 1,)).unique().sort()[0]
    ptr = torch.cat([torch.tensor([0]), ptr, torch.tensor([batch_size])])

    cr = CollisionRate()
    cr.update(pred, trg, ptr)
    cr.compute()


def test_collision_rate_dimension() -> None:
    n_scenarios = 10
    batch_size, seq_len, num_dims = 32, 25, 2
    trg = torch.randn(seq_len, num_dims)
    pred = torch.randn(seq_len, num_dims)

    ptr = torch.randint(1, batch_size, (n_scenarios - 1,)).unique().sort()[0]
    ptr = torch.cat([torch.tensor([0]), ptr, torch.tensor([batch_size])])

    cr = CollisionRate()
    with pytest.raises(AssertionError):
        cr.update(pred, trg, ptr)
