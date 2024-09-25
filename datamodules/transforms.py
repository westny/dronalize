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
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class CoordinateTransform(BaseTransform):
    """
    Transform the coordinates of the agents and map points
     to be relative to the last position of the TA.
    """

    def __call__(self, data: HeteroData) -> HeteroData:
        hist_pos = data['agent']['inp_pos']
        hist_vel = data['agent']['inp_vel']
        hist_acc = data['agent']['inp_acc']
        hist_ori = data['agent']['inp_yaw']

        fut_pos = data['agent']['trg_pos']
        fut_vel = data['agent']['trg_vel']
        fut_ori = data['agent']['trg_yaw']

        map_pos = data['map_point']['position']

        ta_index = data['agent']['ta_index']
        ta_pos = hist_pos[ta_index]
        ta_ori = hist_ori[ta_index]

        # Get the last observed states
        origin = ta_pos[-1].unsqueeze(0)
        ori = ta_ori[-1]

        rot_mat_t = torch.tensor([[torch.cos(ori), -torch.sin(ori)],
                                  [torch.sin(ori), torch.cos(ori)]])

        hist_mask = hist_pos != 0
        fut_mask = fut_pos != 0

        n_hist_pos = (hist_pos - origin) @ rot_mat_t * hist_mask
        n_hist_vel = hist_vel @ rot_mat_t * hist_mask
        n_hist_acc = hist_acc @ rot_mat_t * hist_mask
        n_hist_ori = torch.atan2(torch.sin(hist_ori - ori), torch.cos(hist_ori - ori))

        n_fut_pos = (fut_pos - origin) @ rot_mat_t * fut_mask
        n_fut_vel = fut_vel @ rot_mat_t * fut_mask
        n_fut_ori = torch.atan2(torch.sin(fut_ori - ori), torch.cos(fut_ori - ori))

        n_map_pos = (map_pos - origin) @ rot_mat_t

        data['agent']['inp_pos'] = n_hist_pos
        data['agent']['inp_vel'] = n_hist_vel
        data['agent']['inp_acc'] = n_hist_acc
        data['agent']['inp_yaw'] = n_hist_ori

        data['agent']['trg_pos'] = n_fut_pos
        data['agent']['trg_vel'] = n_fut_vel
        data['agent']['trg_yaw'] = n_fut_ori

        data['map_point']['position'] = n_map_pos

        return data


class CoordinateShift(BaseTransform):
    """
    Shifts the origin of the global coordinate system to be in the last position of the TA.
    """

    def __call__(self, data: HeteroData) -> HeteroData:
        hist_pos = data['agent']['inp_pos']
        fut_pos = data['agent']['trg_pos']
        map_pos = data['map_point']['position']

        ta_index = data['agent']['ta_index']
        ta_pos = hist_pos[ta_index]

        origin = ta_pos[-1].unsqueeze(0)

        hist_mask = hist_pos != 0
        fut_mask = fut_pos != 0

        n_hist_pos = (hist_pos - origin) * hist_mask
        n_fut_pos = (fut_pos - origin) * fut_mask
        n_map_pos = map_pos - origin

        data['agent']['inp_pos'] = n_hist_pos
        data['agent']['trg_pos'] = n_fut_pos
        data['map_point']['position'] = n_map_pos

        return data
