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


class ExpDE(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self,
               pred: torch.Tensor,
               trg: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None,
               logits: bool = False,
               eval_criterion: str = 'FDE',
               mode_first: bool = False) -> None:
        """
        Update the metric state.
        :param: pred: The predicted trajectory. (N, T, M, 2) or (N, M, T, 2)
        :param: trg: The ground-truth target trajectory. (N, T, 2)
        :param: prob: The probability of the predictions. (N, M)
        :param: mask: The mask for valid positions. (N, T)
        :param: logits: Whether the probabilities are logits.
        :param: eval_criterion: Either 'FDE' or 'ADE'.
        :param: mode_first: Whether the mode is the first dimension.
        """

        if eval_criterion not in ['FDE', 'ADE']:
            raise ValueError(f"eval_criterion must be 'FDE' or 'ADE', got {eval_criterion}")

        if pred.dim() != 4:
            raise ValueError(f"pred must be 4-dimensional, got shape {pred.shape}")

        if mask is not None and mask.sum() == 0:
            self.count += 1
            return

        if mode_first:
            # (N, M, T, 2) -> (N, T, M, 2)
            pred = pred.transpose(1, 2)

        batch_size, seq_len, num_modes = pred.size()[:3]

        if prob is None:
            # Uniform distribution if no probabilities are given
            prob = torch.ones(batch_size, pred.shape[2], device=pred.device) / pred.shape[2]  # (N, M)
        elif logits:
            # Convert logits to probabilities
            prob = torch.nn.functional.softmax(prob, dim=-1)  # (N, M)

        trg = trg.unsqueeze(-2).expand(-1, -1, num_modes, -1)  # (N, T, M, 2)

        if eval_criterion == 'ADE':
            norm = torch.linalg.norm(pred - trg, dim=-1)  # (N, T, M)

            if mask is not None:
                num_valid_steps = mask.sum(dim=-1)  # (N,)
                scored_agents = num_valid_steps > 0
                norm = norm * mask.unsqueeze(-1)  # (N, T, M)
                norm = norm[scored_agents]
                num_valid_steps = num_valid_steps[scored_agents]
            else:
                num_valid_steps = torch.ones_like(norm[..., 0]).sum(dim=-1)  # (N,)

            multi_ade = norm.sum(1) / num_valid_steps.unsqueeze(-1)  # (N, M)
            exp_ade = (prob * multi_ade).sum(-1)  # (N,)

            self.sum += exp_ade.sum()
            self.count += exp_ade.size(0)

        elif eval_criterion == 'FDE':

            if mask is not None:
                mask_reversed = 1 * mask.flip(dims=[-1])
                last_idx = seq_len - 1 - mask_reversed.argmax(dim=-1)
                pred = pred[torch.arange(batch_size), last_idx]
                trg = trg[torch.arange(batch_size), last_idx]
                scored_agents = mask.sum(dim=-1) > 0
                pred = pred[scored_agents]
                trg = trg[scored_agents]
            else:
                pred = pred[:, -1]  # (N, M, 2)
                trg = trg[:, -1]  # (N, M, 2)

            multi_fde = torch.linalg.norm(pred - trg, dim=-1)  # (N, M)
            exp_fde = (prob * multi_fde).sum(-1)

            self.sum += exp_fde.sum()
            self.count += exp_fde.size(0)

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric.
        """
        return self.sum / self.count  # type: ignore
