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

from typing import Any
import utm
import torch
import pandas as pd
import osmium as osm
import networkx as nx

edge_types = {
    ("fence", "wall", "road_border"): "road_border",
    ("curbstone",): "curb"
}

type_conversion = {
    None: 1,
    "road_border": 2,
    "curb": 3
}


class OSMHandler(osm.SimpleHandler):
    def __init__(self,
                 utm_x0: float,
                 utm_y0: float,
                 map_x0: float = 0.,
                 map_y0: float = 0.) -> None:
        osm.SimpleHandler.__init__(self)
        self.graph = nx.Graph()
        self.nodes: dict = {}
        self.ways: dict = {}
        self.relations: dict = {}
        self.utm_x0 = utm_x0
        self.utm_y0 = utm_y0
        self.map_x0 = map_x0
        self.map_y0 = map_y0

        self.node_idx = 0

    def coordinate_shift(self, lat: float, lon: float) -> tuple[float, float]:
        utm_x, utm_y, *_ = utm.from_latlon(lat, lon)
        x, y = utm_x - (self.utm_x0 + self.map_x0), utm_y - (self.utm_y0 + self.map_y0)
        return x, y

    def node(self, n: osm.Node) -> None:
        x, y = self.coordinate_shift(n.location.lat, n.location.lon)

        self.graph.add_node(self.node_idx, pos=(x, y), type=None)
        self.nodes[n.id] = self.node_idx
        self.node_idx += 1

    def way(self, w: osm.Way) -> None:
        self.ways[w.id] = {"nodes": [n.ref for n in w.nodes],
                           "tags": {tag.k: tag.v for tag in w.tags}}
        # Add edges to the graph
        nodes = list(w.nodes)
        for i in range(len(nodes) - 1):
            from_node = self.nodes[nodes[i].ref]
            to_node = self.nodes[nodes[i + 1].ref]

            if from_node in self.graph and to_node in self.graph:
                self.graph.add_edge(from_node, to_node, type=None)

        for tag in w.tags:
            if tag.k == 'type':
                for key, value in edge_types.items():
                    if tag.v in key:
                        for i in range(len(nodes) - 1):
                            this_node = self.nodes[nodes[i].ref]
                            next_node = self.nodes[nodes[i + 1].ref]
                            if this_node in self.graph:
                                self.graph.nodes[this_node]['type'] = value
                                if next_node in self.graph:
                                    # check if edge exists
                                    if not self.graph.has_edge(this_node, next_node):
                                        self.graph.add_edge(this_node, next_node, type=value)
                                    else:
                                        # update type
                                        self.graph.edges[this_node, next_node]['type'] = value

    def relation(self, r: osm.Relation) -> None:
        self.relations[r.id] = {"members": [(m.type, m.ref, m.role) for m in r.members],
                                "tags": {tag.k: tag.v for tag in r.tags}}


# @overload
# def get_urban_graph(rec_meta: pd.DataFrame,
#                     lanelet_file: str,
#                     map_x0: float = 0.,
#                     map_y0: float = 0.,
#                     return_torch: bool = True) -> dict:
#     ...
#
#
# @overload
# def get_urban_graph(rec_meta: pd.DataFrame,
#                     lanelet_file: str,
#                     map_x0: float = 0.,
#                     map_y0: float = 0.,
#                     return_torch: bool = False) -> OSMHandler:
#     ...


def get_lanelet_graph(rec_meta: pd.DataFrame,
                      lanelet_file: str,
                      map_x0: float = 0.,
                      map_y0: float = 0.,
                      return_torch: bool = False) -> Any:  # OSMHandler | dict:
    meta = rec_meta
    x_utm_origin = meta.xUtmOrigin.values[0]
    y_utm_origin = meta.yUtmOrigin.values[0]

    osm_handler = OSMHandler(x_utm_origin, y_utm_origin, map_x0, map_y0)
    osm_handler.apply_file(lanelet_file)

    if return_torch:
        return get_torch_graph(osm_handler.graph)

    return osm_handler


def get_torch_graph(graph: nx.Graph) -> dict:
    graph = graph.to_directed(as_view=False)  # this naming convention is the opposite from PyG
    num_nodes = graph.number_of_nodes()

    pos = nx.get_node_attributes(graph, 'pos')
    position = torch.tensor([pos[i] for i in range(num_nodes)], dtype=torch.float)

    node_types = nx.get_node_attributes(graph, 'type')
    node_types = torch.tensor([type_conversion[node_types[i]]
                               for i in range(num_nodes)], dtype=torch.long)

    e_types = nx.get_edge_attributes(graph, 'type')
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([type_conversion[e_types[(u, v)]]
                              for u, v in graph.edges], dtype=torch.long)

    map_data: dict = {
        'map_point': {},
        ('map_point', 'to', 'map_point'): {}
    }

    map_data['map_point']['num_nodes'] = num_nodes
    map_data['map_point']['type'] = node_types
    # map_data['map_point']['driving_dir'] = 0
    # map_data['map_point']['y0'] = 0.0
    map_data['map_point']['position'] = position

    map_data['map_point', 'to', 'map_point']['edge_index'] = edge_index
    map_data['map_point', 'to', 'map_point']['type'] = edge_attr[:, None]

    return map_data


ind_lanelet_mapping = {
    tuple(f"{i:02}" for i in range(7, 10 + 1)): "01_bendplatz",
    tuple(f"{i:02}" for i in range(11, 17 + 1)): "01_bendplatz_construction",
    tuple(f"{i:02}" for i in range(18, 29 + 1)): "02_frankenburg",
    tuple(f"{i:02}" for i in range(30, 32 + 1)): "03_heckstrasse",
    tuple(f"{i:02}" for i in range(0, 6 + 1)): "04_aseag"
}

round_lanelet_mapping = {
    ("00",): "1_kackertstrasse",
    ("01",): "2_thiergarten",
    tuple(f"{i:02}" for i in range(2, 23 + 1)): "0_neuweiler",
}

unid_lanelet_mapping = {
    tuple(f"{i:02}" for i in range(0, 12 + 1)): "0_superc",
}

ds_mapping = {
    "rounD": round_lanelet_mapping,
    "inD": ind_lanelet_mapping,
    "uniD": unid_lanelet_mapping

}

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    ROOT = "../../../data_sets"

    IDX = 30
    DS = "inD"
    STR_IDX = '0' + str(IDX) if IDX < 10 else str(IDX)

    meta_file_pth = f"data/{STR_IDX}_recordingMeta.csv"
    meta_file_pth = os.path.join(ROOT, DS, meta_file_pth)

    meta = pd.read_csv(meta_file_pth)

    PATH = None
    mapping = ds_mapping[DS]

    for key, path in mapping.items():
        if STR_IDX in key:
            PATH = path
            break

    if PATH is None:
        raise ValueError(f"Index {IDX} not found in mapping")

    path_to_lanelet = os.path.join(ROOT, DS, "maps", "lanelets", PATH)
    files = os.listdir(path_to_lanelet)
    path = os.path.join(path_to_lanelet, files[0])

    osm_handler = get_lanelet_graph(meta, path, return_torch=False)

    # # Get all 'relation' unique types
    # unique_types = set()
    # unique_subtypes = set()
    # for way in osm_handler.relations.values():
    #     for k, v in way["tags"].items():
    #         if k == "type":
    #             unique_types.add(v)
    #         else:
    #             unique_subtypes.add(v)
    # print(unique_types)
    # print(unique_subtypes)

    pos = nx.get_node_attributes(osm_handler.graph, 'pos')

    # get all road_border nodes
    road_border_nodes = [n for n, attr in osm_handler.graph.nodes(data=True)
                         if attr['type'] == 'road_border']
    curb_nodes = [n for n, attr in osm_handler.graph.nodes(data=True)
                  if attr['type'] == 'curb']

    # get all edges of type road_border
    road_border_edges = [(u, v) for u, v, attr in osm_handler.graph.edges(data=True)
                         if attr['type'] == 'road_border']
    curb_edges = [(u, v) for u, v, attr in osm_handler.graph.edges(data=True)
                  if attr['type'] == 'curb']
    lane_edges = [(u, v) for u, v, attr in osm_handler.graph.edges(data=True)
                  if attr['type'] is None]

    # get all non road_border nodes
    other = [n for n, attr in osm_handler.graph.nodes(data=True) if attr['type'] is None]

    fig = plt.figure()

    NS = 0.8

    nx.draw_networkx_nodes(osm_handler.graph, pos=pos,
                           nodelist=other, node_color="tab:blue", node_size=NS)
    nx.draw_networkx_nodes(osm_handler.graph, pos=pos,
                           nodelist=curb_nodes, node_color="tab:green", node_size=NS)
    nx.draw_networkx_nodes(osm_handler.graph, pos=pos,
                           nodelist=road_border_nodes, node_color="tab:red", node_size=NS)

    nx.draw_networkx_edges(osm_handler.graph, pos=pos,
                           edgelist=road_border_edges, edge_color='k', width=0.5)
    nx.draw_networkx_edges(osm_handler.graph, pos=pos,
                           edgelist=curb_edges, edge_color='grey', width=0.5)
    nx.draw_networkx_edges(osm_handler.graph, pos=pos,
                           edgelist=lane_edges, edge_color='grey', width=0.5,
                           style='dashed')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.tight_layout()

    bbox = fig.bbox_inches
    bbox_points = [[bbox.x0 + 0.5, bbox.y0 + 1], [bbox.x1 - 0.5, bbox.y1 - 0.8]]
    bbox = Bbox(bbox_points)

    # plt.savefig("in_graph.pdf", bbox_inches=bbox, pad_inches=0)

    plt.show()
