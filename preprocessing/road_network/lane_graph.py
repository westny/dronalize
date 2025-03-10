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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import utm
import osmium as osm
import networkx as nx
import torch
import matplotlib.pyplot as plt

from preprocessing.road_network.edge_type import EdgeType, EDGE_STYLE_MAPPING, NODE_STYLE_MAPPING


@dataclass
class MapPosition:
    """Store map position information."""
    utm_x0: float
    utm_y0: float
    map_x0: float = 0.0
    map_y0: float = 0.0


class LaneGraphBuilder(osm.SimpleHandler):
    """Builds a lane graph from OSM data."""

    def __init__(self, position: MapPosition) -> None:
        super().__init__()
        self.graph = nx.Graph()
        self.nodes: dict[int, int] = {}
        self.ways: dict[int, dict] = {}
        self.relations: dict[int, dict] = {}
        self.position = position
        self.node_idx = 0
        self.zone = self._initialize_utm_zone()

    def _initialize_utm_zone(self) -> Optional[int]:
        """Initialize UTM zone if needed."""
        if self.position.utm_x0 == 0 and self.position.utm_y0 == 0:
            import math
            self.position.utm_x0, self.position.utm_y0, *_ = utm.from_latlon(0, 0)
            return math.floor((0. + 180.) / 6) + 1
        return None

    def coordinate_shift(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert lat/lon to local coordinates."""
        if self.zone is None:
            x, y, *_ = utm.from_latlon(lat, lon)
        else:
            import pyproj
            p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
            x, y = p(lon, lat)

        x -= (self.position.utm_x0 + self.position.map_x0)
        y -= (self.position.utm_y0 + self.position.map_y0)
        return x, y

    def node(self, n: Any) -> None:
        """Process an OSM node."""
        x, y = self.coordinate_shift(n.location.lat, n.location.lon)
        self.graph.add_node(self.node_idx, pos=(x, y), type=EdgeType.NONE)
        self.nodes[n.id] = self.node_idx
        self.node_idx += 1

    def _get_edge_type(self, tags: dict) -> EdgeType:
        """Determine edge type from tags."""
        type_str = tags.get('type')
        subtype = tags.get('subtype')
        return EdgeType.from_str(type_str, subtype)

    def way(self, w: Any) -> None:
        """Process an OSM way."""
        tags = {tag.k: tag.v for tag in w.tags}
        # Use from_str directly instead of _get_edge_type
        edge_type = EdgeType.from_str(tags.get('type'), tags.get('subtype'))

        self.ways[w.id] = {
            "nodes": [n.ref for n in w.nodes],
            "tags": tags,
            "type": edge_type
        }

        nodes = list(w.nodes)

        # Add edges between consecutive nodes
        for i in range(len(nodes) - 1):
            from_node = self.nodes[nodes[i].ref]
            to_node = self.nodes[nodes[i + 1].ref]

            if from_node in self.graph and to_node in self.graph:
                self.graph.add_edge(from_node, to_node, type=edge_type)
                # Update node types based on edge type
                if edge_type != EdgeType.NONE:
                    for node in [from_node, to_node]:
                        current_type = self.graph.nodes[node].get('type')
                        if current_type == EdgeType.NONE:
                            self.graph.nodes[node]['type'] = edge_type

    def relation(self, r: Any) -> None:
        """Process an OSM relation."""
        self.relations[r.id] = {
            "members": [(m.type, m.ref, m.role) for m in r.members],
            "tags": {tag.k: tag.v for tag in r.tags}
        }

        # Handle regulatory elements
        rel_type = self.relations[r.id]["tags"].get('type')
        if rel_type == 'regulatory_element':
            self._process_regulatory_relation(r, EdgeType.REGULATORY)
        elif rel_type == 'stop_line':
            self._process_regulatory_relation(r, EdgeType.STOP)

    def _process_regulatory_relation(self, relation: Any, edge_type: EdgeType) -> None:
        """Process a regulatory relation."""
        for member in relation.members:
            if member.type == "w" and member.role == "ref_line":
                way_nodes = self.ways.get(member.ref, {}).get("nodes", [])
                for i in range(len(way_nodes) - 1):
                    from_node = self.nodes.get(way_nodes[i])
                    to_node = self.nodes.get(way_nodes[i + 1])

                    if from_node in self.graph and to_node in self.graph:
                        for node in [from_node, to_node]:
                            self.graph.nodes[node]['type'] = edge_type
                        self.graph.edges[from_node, to_node]['type'] = edge_type

    def plot(self,
             plot_virtual: bool = True,
             return_axes: bool = False,
             dpi: int = 300,
             save_fig: bool = False) -> None | tuple[plt.Figure, plt.Axes]:
        """Plot the lane graph with enhanced line styles."""
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')

        # Set background color
        fig.patch.set_facecolor('lightgray')  # Change figure background

        pos = nx.get_node_attributes(self.graph, 'pos')

        # Group edges by type
        edges_by_type: dict[EdgeType, list] = {edge_type: [] for edge_type in EdgeType}
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('type', EdgeType.NONE)
            edges_by_type[edge_type].append((u, v))

        # Plot edges with their specific styles
        for edge_type, edges in edges_by_type.items():
            if not plot_virtual and edge_type == EdgeType.VIRTUAL:
                continue
            if not edges:
                continue

            style: dict[str, Any] = EDGE_STYLE_MAPPING[edge_type]
            edge_coords = np.array([(pos[u], pos[v]) for u, v in edges])

            # Separate x and y coordinates
            if len(edge_coords) > 0:
                x_coords = np.array([(pos[u][0], pos[v][0]) for u, v in edges]).T
                y_coords = np.array([(pos[u][1], pos[v][1]) for u, v in edges]).T

                line_style = style.get('style', 'solid')
                if line_style == 'dashed':
                    ax.plot(x_coords, y_coords, color=style['color'],
                            linewidth=style['width'], linestyle='--',
                            dashes=style.get('dashes', [5, 5]),
                            zorder=10)
                elif line_style == 'dotted':
                    ax.plot(x_coords, y_coords, color=style['color'],
                            linewidth=style['width'], linestyle=':',
                            dashes=style.get('dashes', [1, 5]),
                            zorder=10)
                else:
                    ax.plot(x_coords, y_coords, color=style['color'],
                            linewidth=style['width'], linestyle='-',
                            zorder=10)

        plt.axis('off')
        plt.tight_layout()

        if return_axes:
            return fig, ax

        if save_fig:
            plt.savefig('road_network.png', dpi=dpi)
        plt.show()

        return None


def create_torch_graph(graph: nx.Graph) -> dict:
    """Convert NetworkX graph to PyTorch geometric format."""
    directed_graph = graph.to_directed(as_view=False)
    num_nodes = directed_graph.number_of_nodes()

    # Get node positions and types
    pos = nx.get_node_attributes(directed_graph, 'pos')
    position = torch.tensor([pos[i] for i in range(num_nodes)], dtype=torch.float)

    node_types = nx.get_node_attributes(directed_graph, 'type')
    node_type_values = [nt.value for nt in node_types.values()]
    node_types = torch.tensor(node_type_values, dtype=torch.long)

    # Get edge information
    edge_index = torch.tensor(list(directed_graph.edges), dtype=torch.long).t().contiguous()
    edge_types = nx.get_edge_attributes(directed_graph, 'type')
    edge_type_values = [et.value for et in edge_types.values()]
    edge_attr = torch.tensor(edge_type_values, dtype=torch.long)[:, None]

    return {
        'map_point': {
            'num_nodes': num_nodes,
            'type': node_types,
            'position': position
        },
        ('map_point', 'to', 'map_point'): {
            'edge_index': edge_index,
            'type': edge_attr
        }
    }


def plot_torch_graph(
        graph_dict: dict,
        plot_virtual: bool = True,
        node_size: float = 0.2,
        dpi: int = 300,
        plot_nodes: bool = True
) -> None:
    """Plot a lane graph from PyTorch geometric format with enhanced line styles.

    Args:
        graph_dict: Dictionary containing the PyTorch geometric graph data
        plot_virtual: Whether to plot virtual edges
        node_size: Base size multiplier for nodes in the plot
        dpi: DPI of the output plot
        plot_nodes: Whether to plot the nodes
    """
    assert type(graph_dict) == dict, "Make sure get_lane_graph(.) has return_torch=True."

    # Extract node and edge data
    map_point_data = graph_dict['map_point']
    edge_data = graph_dict[('map_point', 'to', 'map_point')]

    positions = map_point_data['position'].numpy()
    node_types = map_point_data['type'].numpy()
    edge_index = edge_data['edge_index'].numpy()
    edge_types = edge_data['type'].squeeze().numpy()

    # Create figure
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    fig.patch.set_facecolor('lightgray')

    # Group edges by type
    edges_by_type: dict[EdgeType, list[tuple[np.ndarray, np.ndarray]]] = {edge_type: [] for edge_type in EdgeType}

    for idx in range(edge_index.shape[1]):
        start_idx = edge_index[0, idx]
        end_idx = edge_index[1, idx]
        edge_type = EdgeType(edge_types[idx])

        # Skip virtual edges if not plotting them
        if not plot_virtual and edge_type == EdgeType.VIRTUAL:
            continue

        edges_by_type[edge_type].append(
            (positions[start_idx], positions[end_idx])
        )

    # Plot edges with their specific styles
    for edge_type, edges in edges_by_type.items():
        if not edges:
            continue

        style = EDGE_STYLE_MAPPING[edge_type]

        # Convert edges to numpy arrays for plotting
        edge_coords = np.array(edges)
        if len(edge_coords) > 0:
            x_coords = np.array([[edge[0][0], edge[1][0]] for edge in edges]).T
            y_coords = np.array([[edge[0][1], edge[1][1]] for edge in edges]).T

            line_style = style.get('style', 'solid')
            if line_style == 'dashed':
                ax.plot(x_coords, y_coords, color=style['color'],
                        linewidth=style['width'], linestyle='--',
                        dashes=style.get('dashes', [5, 5]),
                        zorder=10)
            elif line_style == 'dotted':
                ax.plot(x_coords, y_coords, color=style['color'],
                        linewidth=style['width'], linestyle=':',
                        dashes=style.get('dashes', [1, 5]),
                        zorder=10)
            else:
                ax.plot(x_coords, y_coords, color=style['color'],
                        linewidth=style['width'], linestyle='-',
                        zorder=10)

    # Plot nodes
    if plot_nodes:
        # Group nodes by type
        nodes_by_type: dict[EdgeType, list[np.ndarray]] = {edge_type: [] for edge_type in EdgeType}
        for idx, node_type in enumerate(node_types):
            edge_type = EdgeType(node_type)

            nodes_by_type[edge_type].append(positions[idx])

        # Plot nodes for each type
        for node_type, nodes in nodes_by_type.items():
            if not nodes or (not plot_virtual and node_type == EdgeType.VIRTUAL):
                continue

            node_style: dict[str, Any] = NODE_STYLE_MAPPING[node_type]
            node_arr = np.array(nodes)

            if len(node_arr) > 0:
                ax.scatter(node_arr[:, 0], node_arr[:, 1],
                           c=node_style['color'],
                           s=node_style['size'] * node_size,
                           zorder=20,
                           # edgecolor='black',
                           linewidth=0.5)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_lane_graph(
        lanelet_file: str | bytes,
        utm_x0: float = 0.,
        utm_y0: float = 0.,
        map_x0: float = 0.,
        map_y0: float = 0.,
        return_torch: bool = False
) -> Any:
    """Create a lane graph from a lanelet file."""

    # Ensure lanelet_file is a string (decode if bytes)
    if isinstance(lanelet_file, bytes):
        try:
            lanelet_file = lanelet_file.decode("utf-8")  # Assuming UTF-8 encoded bytes
        except UnicodeDecodeError:
            raise ValueError("lanelet_file is in bytes format but cannot be decoded to a valid string.")

    position = MapPosition(utm_x0, utm_y0, map_x0, map_y0)
    graph_builder = LaneGraphBuilder(position)
    graph_builder.apply_file(lanelet_file)

    return create_torch_graph(graph_builder.graph) if return_torch else graph_builder


# Example usage:
if __name__ == "__main__":
    path = "test_scenario.osm"
    x_utm_origin = 0
    y_utm_origin = 0

    # Create and plot the graph
    graph_builder = get_lane_graph(path, x_utm_origin, y_utm_origin, return_torch=False)
    graph_builder.plot(plot_virtual=True)

    # Create torch graph
    # torch_graph = get_lane_graph(path, x_utm_origin, y_utm_origin, return_torch=True)
    # plot_torch_graph(torch_graph, plot_virtual=True)
