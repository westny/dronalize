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


import torch
from torch import nn
import torch_geometric.nn as ptg
from torch_geometric.data import HeteroData


class Net(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_inputs = config["num_inputs"]
        num_outputs = config["num_outputs"]
        num_hidden = config["num_hidden"]
        self.ph = config["pred_hrz"]

        self.embed = nn.Linear(num_inputs, num_hidden)
        self.encoder = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.interaction = ptg.GATv2Conv(num_hidden, num_hidden, concat=False)
        self.decoder = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.output = nn.Linear(num_hidden, num_outputs)

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_index = data['agent']['edge_index']
        x = torch.cat([data['agent']['inp_pos'],
                       data['agent']['inp_vel'],
                       data['agent']['inp_yaw']], dim=-1)

        # map_to_agent_edge_index = data['map', 'to', 'agent']['edge_index']
        # map_pos = data['map_point']['position']

        x = self.embed(x)
        _, h = self.encoder(x)
        x = h[-1]

        x = self.interaction(x, edge_index)
        x = x.unsqueeze(1).repeat(1, self.ph, 1)
        x, _ = self.decoder(x)

        pred = self.output(x)

        return pred
