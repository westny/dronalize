import pytest
import torch
from metrics.log_likelihood import NegativeLogLikelihood


@pytest.mark.parametrize("dist", ["normal", "mvn", "laplace"])
def test_nll_valid_distribution(dist: str) -> None:
    # Test that no exceptions are raised for valid distributions
    try:
        _ = NegativeLogLikelihood(dist)
    except AssertionError:
        pytest.fail(f"AssertionError raised for valid distribution {dist}")


@pytest.mark.parametrize("dist", ["invalid", "unrecognized", ""])
def test_nll_invalid_distribution(dist: str) -> None:
    # Test that an AssertionError is raised for invalid distributions
    with pytest.raises(ValueError, match=f"Invalid distribution name: {dist}"):
        _ = NegativeLogLikelihood(dist)


def test_nll_mvn_invalid_covariance() -> None:
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)
    logits = torch.randn(batch_size, num_modes)
    prob = torch.softmax(logits, dim=-1)

    # Not square
    scale = torch.ones((batch_size, seq_len, num_modes, num_dims))
    mask = torch.randint(0, 2, (batch_size, seq_len))

    nll = NegativeLogLikelihood("mvn")
    with pytest.raises(AssertionError, match="Covariance matrix must be square."):
        nll.update(pred, trg, scale, prob, mask)


def test_nll_mvn_valid_scale() -> None:
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)
    logits = torch.randn(batch_size, num_modes)
    prob = torch.softmax(logits, dim=-1)
    scale = torch.ones_like(pred).diag_embed() * 0.1
    mask = torch.randint(0, 2, (batch_size, seq_len))

    nll = NegativeLogLikelihood("mvn")
    for msk in [None, mask]:
        nll.update(pred, trg, scale, prob, msk)
        result = nll.compute()
        assert result is not None


@pytest.mark.parametrize("dist", ["normal", "laplace"])
def test_nll_independent(dist: str) -> None:
    batch_size, seq_len, num_modes, num_dims = 32, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)
    logit = torch.randn(batch_size, num_modes)
    scale = torch.ones_like(pred) * 0.1
    mask = torch.randint(0, 2, (batch_size, seq_len))

    nll = NegativeLogLikelihood(dist)
    for lg in [None, logit]:
        nll.update(pred, trg, scale, lg, mask, logits=True)
        result = nll.compute()
        assert result is not None
