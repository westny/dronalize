import torch
import pytest
from metrics.exp_de import ExpDE


def test_exp_de_basic() -> None:
    """Test basic functionality with uniform probabilities."""
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)

    metric = ExpDE()
    # Test both FDE and ADE
    for criterion in ['FDE', 'ADE']:
        metric.update(pred, trg, prob=None, eval_criterion=criterion)
        result = metric.compute()
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # scalar


def test_exp_de_with_mask() -> None:
    """Test with masking of invalid timesteps."""
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))
    prob = torch.ones(batch_size, num_modes) / num_modes  # uniform distribution

    metric = ExpDE()
    for criterion in ['FDE', 'ADE']:
        metric.update(pred, trg, prob, mask=mask, eval_criterion=criterion)
        result = metric.compute()
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result)


def test_exp_de_with_logits() -> None:
    """Test with logits instead of probabilities."""
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)
    logits = torch.randn(batch_size, num_modes)  # unnormalized logits

    metric = ExpDE()
    metric.update(pred, trg, prob=logits, logits=True)
    result = metric.compute()
    assert isinstance(result, torch.Tensor)
    assert not torch.isnan(result)


def test_exp_de_mode_first() -> None:
    """Test with mode dimension before sequence length."""
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, num_modes, seq_len, num_dims)  # note dimension order
    trg = torch.randn(batch_size, seq_len, num_dims)

    metric = ExpDE()
    metric.update(pred, trg, prob=None, mode_first=True)
    result = metric.compute()
    assert isinstance(result, torch.Tensor)
    assert not torch.isnan(result)


def test_exp_de_zero_displacement() -> None:
    """Test with zero displacement between prediction and target."""
    batch_size, seq_len, num_modes, num_dims = 2, 3, 2, 2
    trg = torch.ones(batch_size, seq_len, num_dims)
    pred = torch.ones(batch_size, seq_len, num_modes, num_dims)  # same as target
    prob = torch.ones(batch_size, num_modes) / num_modes

    metric = ExpDE()
    for criterion in ['FDE', 'ADE']:
        metric.update(pred, trg, prob, eval_criterion=criterion)
        result = metric.compute()
        assert result.item() == pytest.approx(0.0)


def test_exp_de_invalid_criterion() -> None:
    """Test that invalid evaluation criterion raises error."""
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)

    metric = ExpDE()
    with pytest.raises(ValueError, match=f"eval_criterion must be 'FDE' or 'ADE', got INVALID"):
        metric.update(pred, trg, prob=None, eval_criterion='INVALID')


def test_exp_de_all_masked() -> None:
    """Test behavior when all timesteps are masked."""
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)
    mask = torch.zeros(batch_size, seq_len)  # all timesteps masked

    metric = ExpDE()
    metric.update(pred, trg, prob=None, mask=mask)
    result = metric.compute()
    assert not torch.isnan(result)


def test_exp_de_accumulation() -> None:
    """Test that metric properly accumulates over multiple updates."""
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)

    metric = ExpDE()
    # Multiple updates
    for _ in range(3):
        metric.update(pred, trg, prob=None)
    result = metric.compute()
    assert isinstance(result, torch.Tensor)
    assert not torch.isnan(result)