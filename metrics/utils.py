# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional
import torch


def filter_prediction(pred: torch.Tensor,
                      trg: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      prob: Optional[torch.Tensor] = None,
                      min_criterion: str = 'FDE',
                      best_idx: Optional[torch.Tensor] = None,
                      mode_first: bool = False
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    if mode_first:
        # (N, M, T, 2) -> (N, T, M, 2)
        pred = pred.transpose(1, 2)

    if pred.size(-1) > 2 or trg.size(-1) > 2:
        warnings.warn("The last dimension of the prediction or target tensors"
                      " is greater than 2. Only the first two dimensions will be considered.")
        pred = pred[..., :2]
        trg = trg[..., :2]

    batch_size, seq_len = pred.size()[:2]

    if best_idx is not None:
        pred = pred[torch.arange(batch_size), :, best_idx]  # (N, T, 2)
        return pred, best_idx

    if min_criterion == "FDE":
        if mask is not None:
            mask_reversed = 1 * mask.flip(dims=[-1])  # (N, T)
            last_idx = seq_len - 1 - mask_reversed.argmax(dim=-1)  # (N,)

            last_pred = pred[torch.arange(batch_size), last_idx]  # (N, M, 2)
            last_trg = trg[torch.arange(batch_size), last_idx]  # (N, 2)
        else:
            last_pred = pred[:, -1]
            last_trg = trg[:, -1]

        best_idx = torch.linalg.norm(last_pred - last_trg.unsqueeze(1),
                                     dim=-1).argmin(dim=-1)  # (N,)

        pred = pred[torch.arange(batch_size), :, best_idx]  # (N, T, 2)

    elif min_criterion == "ADE":
        if mask is not None:
            multi_mask = mask.unsqueeze(-1).unsqueeze(-1)  # (N, T, 1, 1)
            masked_pred = pred * multi_mask  # (N, T, M, 2)
            masked_trg = trg.unsqueeze(2) * multi_mask  # (N, T, 1, 2)
        else:
            masked_pred = pred  # (N, T, M, 2)
            masked_trg = trg.unsqueeze(2)  # (N, T, 1, 2)

        norm = torch.linalg.norm(masked_pred - masked_trg, dim=-1)  # (N, T, M)

        best_idx = norm.sum(dim=1).argmin(dim=-1)  # (N,)
        pred = pred[torch.arange(batch_size), :, best_idx]  # (N, T, 2)

    elif min_criterion == "MAP":
        assert prob is not None, ("Probabilistic criterion requires"
                                  " the probability of the predictions.")

        best_idx = prob.argmax(dim=-1)  # (N,)
        pred = pred[torch.arange(batch_size), :, best_idx]  # (N, T, 2)

    else:
        raise ValueError(f"Invalid criterion: {min_criterion}")

    return pred, best_idx
