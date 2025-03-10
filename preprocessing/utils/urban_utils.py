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
import numpy as np
from pandas import read_csv, concat, merge, DataFrame

from preprocessing.road_network import get_lane_graph
from preprocessing.utils import get_frame_split, compute_acceleration


def add_polar_coordinates(tracks: DataFrame,
                          x0: float = 0.0,
                          y0: float = 0.0) -> DataFrame:
    def polar_to_center(df):
        x, y = df["x"].values, df["y"].values
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        a = np.arctan2(y - y0, x - x0)
        df["rho"], df["theta"] = r, a
        return df

    return tracks.groupby("trackId", group_keys=False).apply(polar_to_center)


def preprocess_levelxd(path: str,
                       rec_id: str,
                       config: dict,
                       output_dir: str,
                       seed: int = 42,
                       dataset: str = "rounD",
                       add_supp: bool = False,
                       debug: bool = False) -> tuple:
    # Get the approximate geographical center of the scene
    p0 = (config["recordings"][rec_id]["x0"], config["recordings"][rec_id]["y0"])

    # Construct the base directory path for your data
    base_dir = os.path.join(path, dataset, "data")

    # Use os.path.join for each specific file
    rec_meta_path = os.path.join(base_dir, f"{rec_id}_recordingMeta.csv")
    tracks_meta_path = os.path.join(base_dir, f"{rec_id}_tracksMeta.csv")
    tracks_path = os.path.join(base_dir, f"{rec_id}_tracks.csv")

    # Read the CSV files
    rec_meta = read_csv(rec_meta_path, engine='pyarrow')
    tracks_meta = read_csv(tracks_meta_path, engine='pyarrow')
    tracks = read_csv(tracks_path, engine='pyarrow')

    # For the lanelet file, construct the path similarly
    location = config["recordings"][rec_id]["location"]
    path_to_lanelet = os.path.join(path, dataset, "maps", "lanelets", location)
    osm_file = os.listdir(path_to_lanelet)[0]
    lanelet_path = os.path.join(path_to_lanelet, osm_file)

    # Get the lanelet graph
    utm_x0 = rec_meta.xUtmOrigin.values[0]
    utm_y0 = rec_meta.yUtmOrigin.values[0]

    lane_graph = get_lane_graph(lanelet_path, utm_x0, utm_y0, p0[0], p0[1], return_torch=True)

    # Perform some initial renaming
    if "vx" not in tracks.columns:
        tracks.rename(columns={"xVelocity": "vx"}, inplace=True)
        tracks.rename(columns={"yVelocity": "vy"}, inplace=True)
        tracks.rename(columns={"xAcceleration": "ax"}, inplace=True)
        tracks.rename(columns={"yAcceleration": "ay"}, inplace=True)
    if "psi" not in tracks.columns:
        tracks.rename(columns={"heading": "psi"}, inplace=True)
        # convert all psi values to radians and wrap to pi
        radians = np.deg2rad(tracks.psi)
        tracks.psi = np.arctan2(np.sin(radians), np.cos(radians))
    if "x" not in tracks.columns:
        tracks.rename(columns={"xCenter": "x"}, inplace=True)
        tracks.rename(columns={"yCenter": "y"}, inplace=True)

    # Shift the data
    tracks.x = tracks.x - p0[0]
    tracks.y = tracks.y - p0[1]

    # If add supp is True, add polar coordinates to the data
    if add_supp:
        tracks['rho'] = 0.
        tracks['theta'] = 0.

        if not debug:
            tracks = add_polar_coordinates(tracks)

    # Make class lowercase in tracks_meta
    tracks_meta["class"] = tracks_meta["class"].str.lower()

    # Determine train, val, test split (by frames)
    train_frames, val_frames, test_frames = get_frame_split(tracks_meta.finalFrame.array[-1], seed=seed)
    frame_dict = {"train": train_frames, "val": val_frames, "test": test_frames}

    shared_args = (rec_id, output_dir, frame_dict, tracks_meta, tracks, lane_graph)

    return shared_args


def preprocess_sind(path: str,
                    rec_id: str,
                    config: dict,
                    output_dir: str,
                    seed: int = 42,
                    dataset: str = "sinD",
                    add_supp: bool = False,
                    debug: bool = False) -> tuple:
    # Get the approximate geographical center of the scene
    p0 = (config["recordings"][rec_id]["x0"], config["recordings"][rec_id]["y0"])

    # Construct the base directory path for your data
    base_dir = path
    content = os.listdir(base_dir)

    for item in content:
        if dataset.lower() == item.lower():
            if config["full"]:
                base_dir = os.path.join(base_dir, item)
            else:
                base_dir = os.path.join(base_dir, item, "Data", rec_id)
            break
    else:
        raise FileNotFoundError(f"Could not find the dataset {dataset} in the path {path}")

    # Get all file paths
    if config["full"]:  # structure should be similar to inD, rounD, ...
        # Lanelet
        lanelet_path = os.path.join(base_dir, "maps", config["recordings"][rec_id]["location"] + ".osm")

        # Get the paths to the pedestrian and vehicle tracks
        ped_tracks_path = os.path.join(base_dir, "data", rec_id, f"Ped_smoothed_tracks.csv")
        veh_tracks_path = os.path.join(base_dir, "data", rec_id, f"Veh_smoothed_tracks.csv")

    else:  # Use directory structure from the SinD GitHub page
        # Lanelet
        lanelet_path = os.path.join(base_dir, config["recordings"][rec_id]["location"] + ".osm")

        # find all folders in the base directory
        folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

        assert len(folders) > 0, "No folders found in the base directory."

        # Get the paths to the pedestrian and vehicle tracks
        ped_tracks_path = os.path.join(base_dir, folders[0], f"Ped_smoothed_tracks.csv")
        veh_tracks_path = os.path.join(base_dir, folders[0], f"Veh_smoothed_tracks.csv")

    # Construct the Lane Graph
    lane_graph = get_lane_graph(lanelet_path, map_x0=p0[0], map_y0=p0[1], return_torch=True)

    # Load the data
    ped_df = read_csv(ped_tracks_path, engine='pyarrow')
    veh_df = read_csv(veh_tracks_path, engine='pyarrow')

    # add heading column to ped_df
    ped_df['psi'] = np.arctan2(ped_df['vy'], ped_df['vx'])

    # add empty columns 'ax', 'ay'
    ped_df['ax'] = 0.
    ped_df['ay'] = 0.

    # Calculate acceleration using numpy gradient
    if not debug:
        ped_df = compute_acceleration(ped_df, 0.1)

    veh_drop_columns = ["yaw_rad", "heading_rad", "length", "width", "v_lon", "v_lat", "a_lon", "a_lat"]
    veh_df = veh_df.drop(columns=veh_drop_columns)

    # Add columns 'psi' to veh_df
    veh_df['psi'] = np.arctan2(veh_df['vy'], veh_df['vx'])

    # Find the maximum track_id in veh_df
    max_veh_id = veh_df['track_id'].max()

    # Get unique track_ids from ped_df
    unique_ped_ids = ped_df['track_id'].unique()

    # Create a mapping from old ped_df track_ids to new track_ids starting from max_veh_id + 1
    mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ped_ids, start=max_veh_id + 1)}

    # Apply this mapping to ped_df
    ped_df['track_id'] = ped_df['track_id'].map(mapping)

    # Merge the two dataframes
    tracks = concat([veh_df, ped_df], ignore_index=True)

    tracks.rename(columns={'agent_type': 'class'}, inplace=True)
    tracks.rename(columns={'track_id': 'trackId'}, inplace=True)
    tracks.rename(columns={'frame_id': 'frame'}, inplace=True)

    # Perform some shifting of the data
    tracks.x = tracks.x - p0[0]
    tracks.y = tracks.y - p0[1]

    # If add supp is True, add polar coordinates to the data
    if add_supp:
        tracks['rho'] = 0.
        tracks['theta'] = 0.

        if not debug:
            tracks = add_polar_coordinates(tracks)

    # get the largest frameId in the dataset
    final_frame = tracks.frame.max()

    # Determine train, val, test split (by frames)
    train_frames, val_frames, test_frames = get_frame_split(final_frame, seed=seed)
    frame_dict = {"train": train_frames, "val": val_frames, "test": test_frames}

    frame_info = tracks.groupby('trackId')['frame'].agg(initialFrame='min', finalFrame='max').reset_index()
    class_info = tracks[['trackId', 'class']].drop_duplicates()

    tracks_meta = merge(class_info, frame_info, on='trackId')

    shared_args = (rec_id, output_dir, frame_dict, tracks_meta, tracks, lane_graph)

    return shared_args
