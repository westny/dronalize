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
import numpy as np
import pandas as pd


def section_graph(lane_markings: list[float],
                  road_x: tuple[float, float],
                  spacing: float = 3.0,
                  direction: int = 1) -> dict:
    """
    Create a graph representation of a section of a highway.
    :param lane_markings: list of y-coordinates of the lane markings
    :param road_x: tuple of the min. and max. x-coordinates of the road
    :param spacing: spacing between the points
    :param direction: direction of the road:
                1 positive right (lower section), -1 positive left (upper section)
    :return:
    """

    pos = []
    node_type = []

    # We would like to have a FLU coordinate system.
    # We move the origin to the bottom right corner of the road
    if direction == 1:
        y0 = lane_markings[0]
    else:
        y0 = lane_markings[-1]

    n_markings = len(lane_markings)

    x_l = np.arange(road_x[0], road_x[1], spacing)
    for j, lmi in enumerate(lane_markings):
        y_l = np.abs(np.ones(x_l.shape) * lmi - y0)
        xi = np.stack((x_l, y_l), axis=1)
        pos.append(xi)
        if j in (0, n_markings - 1):
            node_cls = np.ones_like(x_l) * 2  # 2 is the type for road boundary nodes
        else:
            node_cls = np.ones_like(x_l)  # 1 is the type for lane line nodes
        node_type.append(node_cls)

    pos_arr = np.concatenate(pos, axis=0)
    node_attr = np.concatenate(node_type, axis=0)

    map_data: dict = {
        'map_point': {},
        ('map_point', 'to', 'map_point'): {}
    }

    map_data['map_point']['num_nodes'] = pos_arr.shape[0]
    map_data['map_point']['type'] = torch.from_numpy(node_attr).long()
    # map_data['map_point']['y0'] = y0
    # map_data['map_point']['driving_dir'] = direction
    map_data['map_point']['position'] = torch.from_numpy(pos_arr).float()

    nodes_per_lane = len(x_l)
    node_idx = 0
    edge_index = []
    edge_attr = []
    for j in range(n_markings):
        edges = np.array([[i, i + 1] for i in range(nodes_per_lane - 1)] +
                         [[i + 1, i] for i in range(nodes_per_lane - 1)]).T + node_idx
        edge_index.append(edges)
        node_idx += nodes_per_lane
        if j in (0, n_markings - 1):
            edge_cls = np.ones((len(edges[0]), 1)) * 2  # 2 is the type for road boundaries
        else:
            edge_cls = np.ones((len(edges[0]), 1))  # 1 is the type for lane lines
        edge_attr.append(edge_cls)

    edge_index = np.concatenate(edge_index, axis=1)
    edge_attr = np.concatenate(edge_attr, axis=0)

    map_data['map_point', 'to', 'map_point']['edge_index'] = torch.from_numpy(edge_index).long()
    map_data['map_point', 'to', 'map_point']['type'] = torch.from_numpy(edge_attr).float()

    return map_data


def get_highway_graph(rec_meta: pd.DataFrame,
                      tracks: pd.DataFrame,
                      spacing: float = 3.0,
                      buffer: float = 10.0) -> tuple[dict, dict, float, float]:
    """
    Get the graph representation of the highway from the lane markings.
    :param rec_meta: meta dataframe of the recording (used to get the lane markings)
    :param tracks: trajectory dataframe of the recording (used to get the range of x values)
    :param spacing: spacing between the lane graph nodes
    :param buffer: buffer to add to the range of x values
    :return:
    """

    ulm = [float(l) for l in list(rec_meta['upperLaneMarkings'])[0].split(';')]
    llm = [float(l) for l in list(rec_meta['lowerLaneMarkings'])[0].split(';')]

    x_min = tracks.x.min()
    x_max = tracks.x.max()

    norm_max = int(x_max - x_min) + buffer
    norm_min = - buffer

    # make sure the range is divisible by spacing
    norm_max = norm_max + (spacing - norm_max % spacing)
    norm_min = norm_min - (norm_min % spacing)

    data_ulm = section_graph(ulm, (norm_min, norm_max), spacing, direction=-1)
    data_llm = section_graph(llm, (norm_min, norm_max), spacing, direction=1)
    return data_ulm, data_llm, x_min, x_max


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    i = 4
    rec_idx = f"0{str(i)}" if i < 10 else str(i)
    ROOT = "../../data_sets/highD/data"

    recording_meta = pd.read_csv(f"{ROOT}/{rec_idx}_recordingMeta.csv")
    tracks_csv = pd.read_csv(f"{ROOT}/{rec_idx}_tracks.csv")

    data_upper, data_lower, *_ = get_highway_graph(recording_meta, tracks_csv, spacing=10)

    plot_data = data_upper

    # plt.figure(figsize=(20, 5))

    # plot upper lane markings using edge_index
    for i in range(plot_data['map_point', 'to', 'map_point']['edge_index'].shape[1]):
        source = plot_data['map_point', 'to', 'map_point']['edge_index'][0, i]
        target = plot_data['map_point', 'to', 'map_point']['edge_index'][1, i]
        source_pos = plot_data['map_point']['position'][source]
        target_pos = plot_data['map_point']['position'][target]
        COLOR = 'k' if plot_data['map_point', 'to', 'map_point']['type'][i] == 2 else 'grey'
        plt.plot([source_pos[0], target_pos[0]],
                 [source_pos[1], target_pos[1]], color=COLOR, zorder=1)

    # plot all points
    for i in range(plot_data['map_point']['position'].shape[0]):
        COLOR = 'r' if plot_data['map_point']['type'][i] == 2 else 'b'
        plt.scatter(plot_data['map_point']['position'][i, 0],
                    plot_data['map_point']['position'][i, 1], color=COLOR, s=5, zorder=2)

    # plt.axis('equal')
    plt.show()
