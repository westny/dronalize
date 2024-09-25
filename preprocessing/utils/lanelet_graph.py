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

import numpy as np
import utm
import torch
import pandas as pd
import osmium as osm
import networkx as nx

edge_types = {
    ("fence", "wall", "road_border"): "road_border",
    ("curbstone",): "curb",
    ("stop_line", "regulatory_element"): "regulatory",
    ("virtual",): "virtual",
}

type_conversion = {
    None: 1,
    "road_border": 2,
    "curb": 3,
    "regulatory": 4,
    "virtual": 5,
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

        self.zone = None

        if self.utm_x0 == 0. and self.utm_y0 == 0.:
            import math
            import pyproj

            self.utm_x0, self.utm_y0, *_ = utm.from_latlon(0, 0)
            self.zone = math.floor((0. + 180.) / 6) + 1
            self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')

        self.node_idx = 0

    def from_latlon(self, lat: float, lon: float) -> tuple[float, float]:
        if self.zone is None:
            x, y, *_ = utm.from_latlon(lat, lon)
        else:
            x, y = self.p(lon, lat)
        return x, y

    def coordinate_shift(self, lat: float, lon: float) -> tuple[float, float]:
        utm_x, utm_y = self.from_latlon(lat, lon)
        x, y = utm_x - (self.utm_x0 + self.map_x0), utm_y - (self.utm_y0 + self.map_y0)
        return x, y

    def node(self, n: osm.Node) -> None:  # type: ignore
        x, y = self.coordinate_shift(n.location.lat, n.location.lon)

        self.graph.add_node(self.node_idx, pos=(x, y), type=None)
        self.nodes[n.id] = self.node_idx
        self.node_idx += 1

    def way(self, w: osm.Way) -> None:  # type: ignore
        self.ways[w.id] = {"nodes": [n.ref for n in w.nodes],
                           "tags": {tag.k: tag.v for tag in w.tags}}
        # Add edges to the graph
        nodes = list(w.nodes)
        for i in range(len(nodes) - 1):
            from_node = self.nodes[nodes[i].ref]
            to_node = self.nodes[nodes[i + 1].ref]

            if from_node in self.graph and to_node in self.graph:
                edge_type = None
                if 'type' in w.tags:
                    for key, value in edge_types.items():
                        if w.tags['type'] in key:
                            edge_type = value
                            break
                if 'subtype' in w.tags:
                    for key, value in edge_types.items():
                        if w.tags['subtype'] in key:
                            edge_type = value
                            break

                self.graph.add_edge(from_node, to_node, type=edge_type)

        # Prioritizing node types
        node_type_priority = {"regulatory": 1, "road_border": 2, "curb": 3, "lane_line": 4, "virtual": 5, "other": 6}

        for tag in w.tags:
            if tag.k == 'type':
                for key, value in edge_types.items():
                    if tag.v in key:
                        for i in range(len(nodes) - 1):
                            this_node = self.nodes[nodes[i].ref]
                            next_node = self.nodes[nodes[i + 1].ref]

                            if this_node in self.graph:
                                current_type = self.graph.nodes[this_node].get('type')

                                if (current_type is None or node_type_priority[value] <
                                        node_type_priority[current_type]):
                                    self.graph.nodes[this_node]['type'] = value

                                if next_node in self.graph:
                                    # check if edge exists
                                    if not self.graph.has_edge(this_node, next_node):
                                        self.graph.add_edge(this_node, next_node, type=value)
                                    else:
                                        current_edge_type = self.graph.edges[this_node, next_node].get('type')

                                        if (current_edge_type is None or node_type_priority[value] <
                                                node_type_priority[current_edge_type]):
                                            self.graph.edges[this_node, next_node]['type'] = value

    def relation(self, r: osm.Relation) -> None:  # type: ignore
        self.relations[r.id] = {"members": [(m.type, m.ref, m.role) for m in r.members],
                                "tags": {tag.k: tag.v for tag in r.tags}}

        # Check if the type is 'regulatory_element'
        relation_type = self.relations[r.id]["tags"].get('type')
        if relation_type == 'regulatory_element':
            # Iterate over the members of the relation
            for member in r.members:
                # Only update nodes that are part of this relation
                if member.type == "node":
                    continue
                    node_id = member.ref
                    if node_id in self.nodes:
                        graph_node_id = self.nodes[node_id]
                        # Update the node type to 'stop_line'
                        self.graph.nodes[graph_node_id]['type'] = 'regulatory'

                # You might also want to update the types of edges if necessary
                elif member.type == "w" and member.role == "ref_line":
                    way_id = member.ref
                    if way_id in self.ways:
                        nodes = self.ways[way_id]["nodes"]
                        for i in range(len(nodes) - 1):
                            from_node = self.nodes.get(nodes[i])
                            to_node = self.nodes.get(nodes[i + 1])

                            if from_node in self.graph and to_node in self.graph:
                                # Update the node type
                                self.graph.nodes[from_node]['type'] = 'regulatory'
                                self.graph.nodes[to_node]['type'] = 'regulatory'

                                # Update the edge type to 'stop_line'
                                self.graph.edges[from_node, to_node]['type'] = 'regulatory'

    def finalize_node_types(self) -> None:
        # Assign the remaining node types that are still None
        for node in self.graph.nodes:
            if self.graph.nodes[node]['type'] is None:
                self.graph.nodes[node]['type'] = None

    #         # Set to track which nodes have been visited during the propagation
    #         visited = set()
    #
    #         # Propagate 'road_border' type along all connected components
    #         for node in self.graph.nodes:
    #             if self.graph.nodes[node]['type'] == 'road_border' and node not in visited:
    #                 # Perform BFS or DFS to propagate 'road_border' to all connected nodes
    #                 self.propagate_type(node, 'road_border', visited)
    #
    # def propagate_type(self, start_node: int, node_type: str, visited: set) -> None:
    #     """Use BFS to propagate the node type along all connected nodes."""
    #     queue = [start_node]
    #     visited.add(start_node)
    #
    #     while queue:
    #         node = queue.pop(0)  # BFS: pop from front of the queue
    #         self.graph.nodes[node]['type'] = node_type  # Assign the type to the current node
    #
    #         # Iterate through all neighbors of the current node
    #         for neighbor in self.graph.neighbors(node):
    #             if neighbor not in visited:
    #                 visited.add(neighbor)
    #                 queue.append(neighbor)
    #                 # Set the neighbor's type to 'road_border'
    #                 self.graph.nodes[neighbor]['type'] = node_type
    #
    #                 # Set the edge type to 'road_border'
    #                 self.graph.edges[node, neighbor]['type'] = node_type


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


def get_lanelet_graph(lanelet_file: str,
                      utm_x0: float = 0.,
                      utm_y0: float = 0.,
                      map_x0: float = 0.,
                      map_y0: float = 0.,
                      return_torch: bool = False) -> Any:  # OSMHandler | dict:

    osm_handler = OSMHandler(utm_x0, utm_y0, map_x0, map_y0)
    osm_handler.apply_file(lanelet_file)
    osm_handler.finalize_node_types()

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

exid_lanelet_mapping = {
    tuple(f"{i:02}" for i in range(0, 18 + 1)): "0_cologne_butzweiler",
    tuple(f"{i:02}" for i in range(19, 38 + 1)): "1_cologne_fortiib",
    tuple(f"{i:02}" for i in range(39, 52 + 1)): "2_aachen_brand",
    tuple(f"{i:02}" for i in range(53, 60 + 1)): "3_bergheim_roemer",
    tuple(f"{i:02}" for i in range(61, 72 + 1)): "4_cologne_klettenberg",
    tuple(f"{i:02}" for i in range(73, 77 + 1)): "5_aachen_laurensberg",
    tuple(f"{i:02}" for i in range(78, 92 + 1)): "6_merzenich_rather",
}

ds_mapping = {
    "rounD": round_lanelet_mapping,
    "inD": ind_lanelet_mapping,
    "uniD": unid_lanelet_mapping,
    "exiD": exid_lanelet_mapping,

}

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    ROOT = os.getcwd()
    while "datasets" not in os.listdir(ROOT):
        ROOT = os.path.join(ROOT, "..")
    ROOT = os.path.join(ROOT, "datasets")

    PLOT_VIRTUAL = True
    PLOT_TRACKS = False

    # IDX = 30
    # DS = "inD"

    # IDX = 1
    # DS = "rounD"

    # IDX = 2
    # DS = "sinD"

    IDX = 0
    DS = "uniD"

    # IDX = 79  # 53 # 39 # 19 # 79 #19 # 0
    # DS = "exiD"

    STR_IDX = '0' + str(IDX) if IDX < 10 else str(IDX)

    meta_file_pth = f"data/{STR_IDX}_recordingMeta.csv"
    meta_file_pth = os.path.join(ROOT, DS, meta_file_pth)
    meta = pd.read_csv(meta_file_pth)
    x_utm_origin = meta.xUtmOrigin.values[0]
    y_utm_origin = meta.yUtmOrigin.values[0]

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

    osm_handler = get_lanelet_graph(path, x_utm_origin, y_utm_origin, return_torch=False)

    pos = nx.get_node_attributes(osm_handler.graph, 'pos')

    # get all road_border nodes
    road_border_nodes = [n for n, attr in osm_handler.graph.nodes(data=True)
                         if attr['type'] == 'road_border']
    curb_nodes = [n for n, attr in osm_handler.graph.nodes(data=True)
                  if attr['type'] == 'curb']
    virtual_nodes = [n for n, attr in osm_handler.graph.nodes(data=True)
                     if attr['type'] == 'virtual']

    # get all edges of type road_border
    road_border_edges = [(u, v) for u, v, attr in osm_handler.graph.edges(data=True)
                         if attr['type'] == 'road_border']
    curb_edges = [(u, v) for u, v, attr in osm_handler.graph.edges(data=True)
                  if attr['type'] == 'curb']
    lane_edges = [(u, v) for u, v, attr in osm_handler.graph.edges(data=True)
                  if attr['type'] is None]
    virtual_edges = [(u, v) for u, v, attr in osm_handler.graph.edges(data=True)
                     if attr['type'] == 'virtual']

    # get all pedestrian_marking nodes
    stop_nodes = [n for n, attr in osm_handler.graph.nodes(data=True)
                  if attr['type'] == 'regulatory']

    # get all pedestrian_marking edges
    stop_edges = [(u, v) for u, v, attr in osm_handler.graph.edges(data=True)
                  if attr['type'] == 'regulatory']

    # get all non road_border nodes
    other = [n for n, attr in osm_handler.graph.nodes(data=True) if attr['type'] is None]

    fig = plt.figure(dpi=300)

    NS = 0.8 if DS == "inD" or DS == "uniD" else 0.2

    nx.draw_networkx_nodes(osm_handler.graph, pos=pos,
                           nodelist=other, node_color="tab:blue", node_size=NS)
    nx.draw_networkx_nodes(osm_handler.graph, pos=pos,
                           nodelist=curb_nodes, node_color="tab:green", node_size=NS)
    nx.draw_networkx_nodes(osm_handler.graph, pos=pos,
                           nodelist=road_border_nodes, node_color="tab:red", node_size=NS)
    nx.draw_networkx_nodes(osm_handler.graph, pos=pos, nodelist=stop_nodes, node_color="tab:orange", node_size=NS)

    nx.draw_networkx_edges(osm_handler.graph, pos=pos,
                           edgelist=road_border_edges, edge_color='k', width=0.5)
    nx.draw_networkx_edges(osm_handler.graph, pos=pos,
                           edgelist=curb_edges, edge_color='grey', width=0.5)
    nx.draw_networkx_edges(osm_handler.graph, pos=pos,
                           edgelist=lane_edges, edge_color='grey', width=0.5,
                           style='dashed')
    nx.draw_networkx_edges(osm_handler.graph, pos=pos, edgelist=stop_edges, edge_color='tab:orange', width=0.5)

    if PLOT_VIRTUAL:
        nx.draw_networkx_nodes(osm_handler.graph, pos=pos, nodelist=virtual_nodes, node_color="tab:blue", node_size=NS)
        nx.draw_networkx_edges(osm_handler.graph, pos=pos, edgelist=virtual_edges, edge_color='grey', width=0.7,
                               style="dotted")

    if PLOT_TRACKS:
        tracks_file_pth = f"data/{STR_IDX}_tracks.csv"
        tracks_file_pth = os.path.join(ROOT, DS, tracks_file_pth)
        tracks = pd.read_csv(tracks_file_pth)

        track_ids = tracks.trackId.unique()
        track_ids = np.random.choice(track_ids, 100)
        for tid in track_ids:
            df1 = tracks[tracks.trackId == tid]
            pos = df1[["xCenter", "yCenter"]].values
            plt.plot(pos[:, 0], pos[:, 1], lw=1, c='k', zorder=0, alpha=1.)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.tight_layout()

    bbox = fig.bbox_inches
    bbox_points = [[bbox.x0 + 0.5, bbox.y0 + 1], [bbox.x1 - 0.5, bbox.y1 - 0.8]]
    bbox = Bbox(bbox_points)

    plt.savefig("unid_graph.svg", bbox_inches=bbox, pad_inches=0)

    plt.show()
