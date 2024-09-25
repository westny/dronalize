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
from typing import Optional, Any
import torch
import torch.distributions as tdist
from torchmetrics import Metric


class NegativeLogLikelihood(Metric):
    dist: Any

    def __init__(self,
                 dist: str = "mvn",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        self.dist = self.get_distribution_initializer(dist)

    @staticmethod
    def get_distribution_initializer(dist_name: str) -> Any:
        if dist_name == "mvn":
            return tdist.MultivariateNormal
        if dist_name == "normal":
            return tdist.Normal
        if dist_name == "laplace":
            return tdist.Laplace
        raise ValueError(f"Invalid distribution name: {dist_name}")

    @staticmethod
    def handle_mode_first(pred: torch.Tensor, scale: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred.dim() == 4:
            return pred.transpose(1, 2), scale.transpose(1, 2)
        warnings.warn("'mode_first' is set to True but the predictions"
                      " are not multi-modal. Ignoring the flag.")
        return pred, scale

    def create_distribution(self, pred, scale, is_tril):
        if self.dist.__name__ == "MultivariateNormal":
            assert scale.size(-1) == scale.size(-2), "Covariance matrix must be square."
            if not is_tril:
                scale = torch.linalg.cholesky(scale)
            return self.dist(loc=pred, scale_tril=scale)
        return self.dist(loc=pred, scale=scale)

    def update(self,
               pred: torch.Tensor,
               trg: torch.Tensor,
               scale: torch.Tensor,
               prob: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None,
               logits: bool = False,
               is_tril: bool = False,
               mode_first: bool = False) -> None:
        """
        Update the metric state.
        :param: pred: The predicted trajectory. (N, T, M, 2) or (N, T, 2)
        :param: trg: The ground-truth target trajectory. (N, T, 2)
        :param: scale: The scale of the predictions. (N, T, M, 2, (2)) or (N, T, 2, (2))
        :param: prob: The probability of the predictions. (N, M)
        :param: mask: The mask for valid positions. (N, T)
        :param: logits: Whether the probabilities are logits.
        :param: is_tril: Whether the scale is a lower triangular matrix.
        :param: mode_first: Whether the mode is the first dimension.
        """

        if mode_first:
            # (N, M, T, 2) -> (N, T, M, 2)
            pred, scale = self.handle_mode_first(pred, scale)

        batch_size, seq_len = pred.size()[:2]

        distribution = self.create_distribution(pred, scale, is_tril)

        if pred.dim() == 4:
            if prob is None:
                prob = torch.ones(batch_size, pred.shape[2], device=pred.device) / pred.shape[2]
                if logits:
                    prob *= 0.0
            prob = prob.unsqueeze(1).expand(-1, seq_len, -1)  # (N, T, M)

            mix = tdist.Categorical(logits=prob) if logits else tdist.Categorical(probs=prob)
            if self.dist.__name__ != "MultivariateNormal":
                distribution = tdist.Independent(distribution, 1)
            distribution = tdist.MixtureSameFamily(mix, distribution)

        # Compute the negative log-likelihood
        neg_log_prob = distribution.log_prob(trg).neg()  # (N, T)

        if mask is not None:
            neg_log_prob = neg_log_prob * mask
            valid_time_steps = mask.sum(dim=-1)
            scored_agents = valid_time_steps > 0
            neg_log_prob = neg_log_prob[scored_agents]
            valid_time_steps = valid_time_steps[scored_agents]
        else:
            valid_time_steps = torch.ones_like(neg_log_prob).sum(-1)  # (N,)

        nll = neg_log_prob.sum(-1) / valid_time_steps  # (N,)

        self.sum += nll.sum()
        self.count += nll.size(0)

    def compute(self) -> torch.Tensor:
        """
        Compute the final metric.
        """
        return self.sum / self.count  # type: ignore
