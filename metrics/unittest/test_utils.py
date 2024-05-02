import pytest
import torch
from metrics.utils import filter_prediction


def test_filter_prediction_pre_indexed() -> None:
    batch_size, seq_len, num_modes, num_dims = 10, 25, 6, 2
    trg = torch.randn(batch_size, seq_len, num_dims)

    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    best_idx = torch.randint(0, num_modes, (batch_size,))
    pred_, best_idx_ = filter_prediction(pred, trg, best_idx=best_idx)
    assert pred_.size() == (batch_size, seq_len, num_dims)
    assert (best_idx == best_idx_).all()


def test_filter_prediction_criterion() -> None:
    batch_size, seq_len, num_modes, num_dims = 10, 25, 6, 2
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))
    prob = torch.randn(batch_size, num_modes)

    for min_criterion in ['FDE', 'ADE', 'MAP']:
        pred_, best_idx_ = filter_prediction(pred, trg, mask, prob,
                                             min_criterion, mode_first=False)
        assert pred_.size() == (batch_size, seq_len, num_dims)
        assert best_idx_.size() == (batch_size,)

    invalid_criterion = 'invalid'
    with pytest.raises(ValueError) as exc:
        filter_prediction(pred, trg, min_criterion=f'{invalid_criterion}')

        assert exc.value == f"Invalid criterion: {invalid_criterion}"


def test_filter_prediction_mode_consistency() -> None:
    batch_size, seq_len, num_modes, num_dims = 10, 25, 6, 2
    trg = torch.randn(batch_size, seq_len, num_dims)
    mask = torch.randint(0, 2, (batch_size, seq_len))
    prob = torch.randn(batch_size, num_modes)

    for min_criterion in ['FDE', 'ADE', 'MAP']:
        pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
        pred_list = []
        best_idx_list = []
        for mode_first in [False, True]:
            if mode_first:
                pred_ = pred.transpose(1, 2).clone()
            else:
                pred_ = pred.clone()

            pred_, best_idx_ = filter_prediction(pred_, trg, mask, prob,
                                                 min_criterion, mode_first=mode_first)

            pred_list.append(pred_)
            best_idx_list.append(best_idx_)

        assert torch.allclose(pred_list[0], pred_list[1])
        assert torch.allclose(best_idx_list[0], best_idx_list[1])


def test_filter_prediction_dimension() -> None:
    batch_size, seq_len, num_modes, num_dims = 10, 25, 6, 4
    pred = torch.randn(batch_size, seq_len, num_modes, num_dims)
    trg = torch.randn(batch_size, seq_len, num_dims)
    with pytest.warns(UserWarning):
        pred_, best_idx_ = filter_prediction(pred, trg)
        assert pred_.size() == (batch_size, seq_len, 2)
        assert best_idx_.size() == (batch_size,)
