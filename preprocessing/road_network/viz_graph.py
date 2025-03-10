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

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing.road_network import get_lane_graph

# Dataset mappings
DATASET_MAPPINGS = {
    "inD": {
        tuple(f"{i:02}" for i in range(7, 10 + 1)): "01_bendplatz",
        tuple(f"{i:02}" for i in range(11, 17 + 1)): "01_bendplatz_construction",
        tuple(f"{i:02}" for i in range(18, 29 + 1)): "02_frankenburg",
        tuple(f"{i:02}" for i in range(30, 32 + 1)): "03_heckstrasse",
        tuple(f"{i:02}" for i in range(0, 6 + 1)): "04_aseag"
    },
    "rounD": {
        ("00",): "1_kackertstrasse",
        ("01",): "2_thiergarten",
        tuple(f"{i:02}" for i in range(2, 23 + 1)): "0_neuweiler",
    },
    "uniD": {
        tuple(f"{i:02}" for i in range(0, 12 + 1)): "0_superc",
    },
    "exiD": {
        tuple(f"{i:02}" for i in range(0, 18 + 1)): "0_cologne_butzweiler",
        tuple(f"{i:02}" for i in range(19, 38 + 1)): "1_cologne_fortiib",
        tuple(f"{i:02}" for i in range(39, 52 + 1)): "2_aachen_brand",
        tuple(f"{i:02}" for i in range(53, 60 + 1)): "3_bergheim_roemer",
        tuple(f"{i:02}" for i in range(61, 72 + 1)): "4_cologne_klettenberg",
        tuple(f"{i:02}" for i in range(73, 77 + 1)): "5_aachen_laurensberg",
        tuple(f"{i:02}" for i in range(78, 92 + 1)): "6_merzenich_rather",
    }
}


def find_root_directory() -> str:
    """Find the root directory containing the datasets folder."""
    root = os.getcwd()
    while "datasets" not in os.listdir(root):
        root = os.path.join(root, "..")
    return os.path.join(root, "datasets")


def get_path_for_index(dataset: str, idx: int) -> Optional[str]:
    """Get the corresponding path for a given dataset and index."""
    str_idx = f"0{idx}" if idx < 10 else str(idx)
    mapping = DATASET_MAPPINGS.get(dataset)

    if not mapping:
        return None

    for key, path in mapping.items():
        if str_idx in key:
            return path
    return None


def load_meta_data(root_dir: str, dataset: str, idx: int) -> tuple[float, float]:
    """Load meta data for the given dataset and index."""
    str_idx = f"0{idx}" if idx < 10 else str(idx)
    meta_file_path = os.path.join(root_dir, dataset, "data", f"{str_idx}_recordingMeta.csv")
    meta = pd.read_csv(meta_file_path)
    return meta.xUtmOrigin.values[0], meta.yUtmOrigin.values[0]


def plot_tracks(ax: plt.Axes, tracks_path: str, num_samples: int = 100) -> None:
    """Plot random track samples on the given axes."""
    tracks = pd.read_csv(tracks_path)
    track_ids = np.random.choice(tracks.trackId.unique(), num_samples)

    for tid in track_ids:
        df = tracks[tracks.trackId == tid]
        pos = df[["xCenter", "yCenter"]].values
        ax.plot(pos[:, 0], pos[:, 1], lw=1, c='tab:purple', zorder=0, alpha=1.)


def main():
    # Configuration
    plot_config = {
        "plot_virtual": True,
        "plot_tracks": True,
        "dataset": "inD",
        "index": 0,
        "num_track_samples": 10
    }

    # Find root directory and setup paths
    root_dir = find_root_directory()

    # Get path for the given dataset and index
    path = get_path_for_index(plot_config["dataset"], plot_config["index"])
    if not path:
        raise ValueError(f"Index {plot_config['index']} not found in mapping for dataset {plot_config['dataset']}")

    # Load meta data
    x_utm_origin, y_utm_origin = load_meta_data(
        root_dir,
        plot_config["dataset"],
        plot_config["index"]
    )

    # Find and load lanelet file
    lanelet_dir = os.path.join(root_dir, plot_config["dataset"], "maps", "lanelets", path)
    lanelet_file = os.path.join(lanelet_dir, os.listdir(lanelet_dir)[0])

    # Create and plot graph
    graph_builder = get_lane_graph(
        lanelet_file,
        x_utm_origin,
        y_utm_origin,
        return_torch=False
    )
    fig, ax = graph_builder.plot(
        plot_virtual=plot_config["plot_virtual"],
        return_axes=True
    )

    # Plot tracks if requested
    if plot_config["plot_tracks"]:
        str_idx = f"0{plot_config['index']}" if plot_config['index'] < 10 else str(plot_config['index'])
        tracks_path = os.path.join(
            root_dir,
            plot_config["dataset"],
            "data",
            f"{str_idx}_tracks.csv"
        )
        plot_tracks(ax, tracks_path, plot_config["num_track_samples"])

    plt.show()


if __name__ == "__main__":
    main()
