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

from math import comb
from typing import Optional
import torch
from torchmetrics import Metric
from metrics.utils import filter_prediction


class CollisionRate(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               trg: torch.Tensor,
               ptr: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None,
               best_idx: Optional[torch.Tensor] = None,
               collision_criterion: str = 'FDE',
               collision_threshold: float = 1.0,
               mode_first: bool = False) -> None:
        """
        Update the metric state.
        :param: pred: The predicted trajectory. (N, T, M, 2) or (N, T, 2)
        :param: trg: The ground-truth target trajectory. (N, T, 2)
        :param: ptr: The pointer tensor to indicate which agents are in the same scene. (batch_size)
        :param: prob: The probability of the predictions. (N, M)
        :param: mask: The mask for valid positions. (N, T)
        :param: best_idx: The index of the best prediction. (N,) (to avoid recomputing it)
        :param: collision_criterion: Either 'FDE', 'ADE', or 'MAP'.
        :param: collision_threshold: The collision threshold in meters.
        :param: mode_first: Whether the mode is the first dimension. (default: False)
        """

        assert pred.dim() > 2, "The prediction tensor must have at least 3 dimensions."

        if pred.dim() == 4:
            pred, _ = filter_prediction(pred, trg, mask, prob, collision_criterion,
                                        best_idx, mode_first=mode_first)

        seq_len = pred.size(1)

        # Compute the collision rate for each scenario
        for i in range(len(ptr) - 1):
            ptr_from = ptr[i]
            ptr_to = ptr[i + 1]

            # Get the scenario
            scenario = pred[ptr_from:ptr_to]
            n = scenario.size(0)

            # Compute the number of possible collisions
            self.count += seq_len * comb(n, 2)  # type: ignore  # T * (n * (n - 1)) // 2
            for t in range(seq_len):
                dists = torch.cdist(scenario[:, t], scenario[:, t], p=2)  # (n, n)

                # Find the collisions and filter out the self-collisions
                collisions = (dists < collision_threshold) & (dists != 0.0)
                self.sum += collisions.sum().item() / 2  # type: ignore

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric.
        """
        return self.sum / self.count  # type: ignore
