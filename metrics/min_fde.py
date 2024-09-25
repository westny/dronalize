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

from typing import Optional
import torch
from torchmetrics import Metric
from metrics.utils import filter_prediction


class MinFDE(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               trg: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None,
               best_idx: Optional[torch.Tensor] = None,
               min_criterion: str = 'FDE',
               mode_first: bool = False) -> None:
        """
        Update the metric state.
        :param: pred: The predicted trajectory. (N, T, M, 2) or (N, T, 2)
        :param: trg: The ground-truth target trajectory. (N, T, 2)
        :param: prob: The probability of the predictions. (N, M)
        :param: mask: The mask for valid positions. (N, T)
        :param: best_idx: The index of the best prediction. (N,) (to avoid recomputing it)
        :param: min_criterion: Either 'FDE', 'ADE', or 'MAP'.
        :param: mode_first: Whether the mode is the first dimension. (default: False)
        """

        if pred.dim() == 4:
            pred, _ = filter_prediction(pred, trg, mask, prob, min_criterion,
                                        best_idx, mode_first=mode_first)

        batch_size, seq_len = pred.size()[:2]

        if mask is not None:
            mask_reversed = 1 * mask.flip(dims=[-1])
            last_idx = seq_len - 1 - mask_reversed.argmax(dim=-1)

            pred = pred[torch.arange(batch_size), last_idx]  # (N, 2)
            trg = trg[torch.arange(batch_size), last_idx]  # (N, 2)

            scored_agents = mask.sum(dim=-1) > 0
            pred = pred[scored_agents]
            trg = trg[scored_agents]
        else:
            pred = pred[:, -1]  # (N, 2)
            trg = trg[:, -1]  # (N, 2)

        fde = torch.linalg.norm(pred - trg, dim=-1)  # (N,)

        self.sum += fde.sum()
        self.count += fde.size(0)

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric.
        """
        return self.sum / self.count  # type: ignore
