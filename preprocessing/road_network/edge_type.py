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


from enum import Enum
from typing import Optional


class EdgeType(Enum):
    """Enumeration of edge types in the lane graph."""
    NONE = 1
    ROAD_BORDER = 2
    CURB = 3
    REGULATORY = 4
    VIRTUAL = 5
    LINE_THIN = 6
    LINE_THIN_DASHED = 7
    LINE_THICK = 8
    LINE_THICK_DASHED = 9
    PEDESTRIAN_MARKING = 10
    BIKE_MARKING = 11
    GUARD_RAIL = 12
    STOP = 13

    @classmethod
    def from_str(cls, type_str: Optional[str], subtype: Optional[str] = None) -> 'EdgeType':
        """Convert string representation to EdgeType."""
        # First check high-priority mappings
        type_map = {
            None: cls.NONE,
            'road_border': cls.ROAD_BORDER,
            'curbstone': cls.CURB,
            'stop_line': cls.STOP,
            'regulatory_element': cls.REGULATORY,
            'virtual': cls.VIRTUAL,
            'pedestrian_marking': cls.PEDESTRIAN_MARKING,
            'bike_marking': cls.BIKE_MARKING,
            'guard_rail': cls.GUARD_RAIL,
            'fence': cls.ROAD_BORDER,
            'wall': cls.ROAD_BORDER
        }

        # Check if it's a high-priority type first
        edge_type = type_map.get(type_str)
        if edge_type is not None:
            return edge_type

        # Only then check for line types
        if type_str == "line_thin":
            return cls.LINE_THIN_DASHED if subtype == "dashed" else cls.LINE_THIN
        elif type_str == "line_thick":
            return cls.LINE_THICK_DASHED if subtype == "dashed" else cls.LINE_THICK

        return cls.NONE


EDGE_STYLE_MAPPING = {
    EdgeType.NONE: {'color': 'grey', 'width': 0.5, 'style': 'dashed'},
    EdgeType.ROAD_BORDER: {'color': 'black', 'width': 1.0, 'style': 'solid'},
    EdgeType.CURB: {'color': 'black', 'width': 1.0, 'style': 'solid'},
    EdgeType.REGULATORY: {'color': 'tab:orange', 'width': 1.0, 'style': 'solid'},
    EdgeType.VIRTUAL: {'color': 'tab:blue', 'width': 1.0, 'style': 'dotted', 'dashes': [2, 5]},
    EdgeType.LINE_THIN: {'color': 'white', 'width': 1.0, 'style': 'solid'},
    EdgeType.LINE_THIN_DASHED: {'color': 'white', 'width': 1.0, 'style': 'dashed', 'dashes': [10, 10]},
    EdgeType.LINE_THICK: {'color': 'white', 'width': 2.0, 'style': 'solid'},
    EdgeType.LINE_THICK_DASHED: {'color': 'white', 'width': 2.0, 'style': 'dashed', 'dashes': [10, 10]},
    EdgeType.PEDESTRIAN_MARKING: {'color': 'white', 'width': 1.0, 'style': 'dashed', 'dashes': [5, 10]},
    EdgeType.BIKE_MARKING: {'color': 'white', 'width': 1.0, 'style': 'dashed', 'dashes': [5, 10]},
    EdgeType.GUARD_RAIL: {'color': 'black', 'width': 1.0, 'style': 'solid'},
    EdgeType.STOP: {'color': 'red', 'width': 1.0, 'style': 'solid'},
}

NODE_STYLE_MAPPING = {
    EdgeType.NONE: {'color': 'grey', 'size': 20},
    EdgeType.ROAD_BORDER: {'color': 'tab:red', 'size': 20},
    EdgeType.CURB: {'color': 'tab:green', 'size': 20},
    EdgeType.REGULATORY: {'color': 'tab:orange', 'size': 30},
    EdgeType.VIRTUAL: {'color': 'tab:blue', 'size': 20},
    EdgeType.LINE_THIN: {'color': 'white', 'size': 20},
    EdgeType.LINE_THIN_DASHED: {'color': 'white', 'size': 20},
    EdgeType.LINE_THICK: {'color': 'white', 'size': 25},
    EdgeType.LINE_THICK_DASHED: {'color': 'white', 'size': 25},
    EdgeType.PEDESTRIAN_MARKING: {'color': 'yellow', 'size': 20},
    EdgeType.BIKE_MARKING: {'color': 'cyan', 'size': 25},
    EdgeType.GUARD_RAIL: {'color': 'black', 'size': 20},
    EdgeType.STOP: {'color': 'red', 'size': 25},
}
