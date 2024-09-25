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
from pandas import DataFrame, unique, read_csv


def add_driving_direction_exits(tracks_meta: DataFrame,
                                tracks: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Add driving direction (1 or 2) as a feature to the tracks dataframe.
    If driving direction is 1, the vehicle is driving from right to left (negative x).
    If driving direction is 2, the vehicle is driving from left to right (positive x).
    """

    # Initialize the drivingDirection columns with empty values
    tracks['drivingDirection'] = np.empty(len(tracks))
    tracks_meta['drivingDirection'] = np.empty(len(tracks_meta))

    # Define a function to process each trackId group
    def process_group(df):
        # Determine driving direction based on the first and last x-values
        x = df['x'].to_numpy()
        if x[0] < x[-1]:
            driving_direction = 2
        else:
            driving_direction = 1

        # Assign the driving direction to the entire group
        df['drivingDirection'] = driving_direction
        return df

    # Apply the driving direction calculation to the tracks DataFrame
    tracks = tracks.groupby('trackId').apply(process_group)

    # Reset index to maintain DataFrame structure
    tracks = tracks.reset_index(drop=True)

    # Update the drivingDirection in tracks_meta based on tracks
    def update_tracks_meta(df):
        # Get the driving direction for the current trackId from the tracks DataFrame
        driving_direction = tracks[tracks['trackId'] == df['trackId'].iloc[0]]['drivingDirection'].iloc[0]
        df['drivingDirection'] = driving_direction
        return df

    # Apply the update to the tracks_meta DataFrame
    tracks_meta = tracks_meta.groupby('trackId').apply(update_tracks_meta)
    tracks_meta = tracks_meta.reset_index(drop=True)

    return tracks_meta, tracks


def add_maneuver_exits(tracks: DataFrame,
                       fz: int = 25,
                       debug: bool = False) -> DataFrame:
    """
    Add maneuver as a feature to the tracks dataframe.

    There are 7 different maneuvers:
    0: left lane change within the next (1) second
    1: left lane change within the next 3 seconds
    2: left lane change within the next 5 seconds
    3: no lane change (lane keep)
    4: right lane change within the next (1) second
    5: right lane change within the next 3 seconds
    6: right lane change within the next 5 seconds
    """

    lane_change_left = [0, 1, 2]
    lane_change_right = [4, 5, 6]

    # Initialize maneuver column with 3 (no lane change)
    tracks['maneuver'] = np.ones(len(tracks), dtype=int) * 3

    if debug:
        return tracks

    # Define a function to process each group of trackId
    def process_group(df):
        if df['laneChange'].nunique() > 1:
            dy = df['latLaneCenterOffset'].to_numpy()
            frames = df['frame'].to_numpy()
            lane_change = df['laneChange'].to_numpy()

            # Detect lane change event indices
            event_indices = [i for i in range(1, len(lane_change)) if
                             lane_change[i] == 1 and lane_change[i - 1] == 0]

            for event_index in event_indices:
                five_seconds_prior = range(max(0, event_index - 5 * fz + 1), event_index)
                three_seconds_prior = range(max(0, event_index - 3 * fz + 1), event_index)
                one_second_prior = range(max(0, event_index - fz + 1), event_index)

                five_second_frames = (frames[i] for i in five_seconds_prior)
                three_second_frames = (frames[i] for i in three_seconds_prior)
                one_second_frames = (frames[i] for i in one_second_prior)

                ddelta_y = dy[event_index] - dy[event_index - 1]
                maneuvers = lane_change_right if ddelta_y > 0 else lane_change_left

                # Assign maneuvers based on the prior frames
                df.loc[df['frame'].isin(five_second_frames), 'maneuver'] = maneuvers[2]
                df.loc[df['frame'].isin(three_second_frames), 'maneuver'] = maneuvers[1]
                df.loc[df['frame'].isin(one_second_frames), 'maneuver'] = maneuvers[0]

        return df

    # Group by trackId and apply the maneuver processing to each group
    tracks = tracks.groupby('trackId').apply(process_group)

    # Reset index to maintain DataFrame structure
    tracks = tracks.reset_index(drop=True)

    return tracks


def preprocess_exid(path: str,
                    rec_id: str,
                    config: dict,
                    output_dir: str,
                    seed: int = 42,
                    dataset: str = "exiD",
                    debug: bool = False) -> tuple:
    from preprocessing.utils.lanelet_graph import get_lanelet_graph
    from preprocessing.utils.common import get_frame_split

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

    lane_graph = get_lanelet_graph(lanelet_path, utm_x0, utm_y0, p0[0], p0[1], return_torch=True)

    lane_graph = {'upper_map': lane_graph, 'lower_map': lane_graph}

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
        tracks.x = tracks.x - p0[0]
        tracks.y = tracks.y - p0[1]

    tracks_meta, tracks = add_driving_direction_exits(tracks_meta, tracks)
    tracks = add_maneuver_exits(tracks, debug=debug)
    tracks_meta["numLaneChanges"] = tracks.groupby("trackId")["laneChange"].sum().values


    # Determine train, val, test split (by frames)
    train_frames, val_frames, test_frames = get_frame_split(tracks_meta.finalFrame.array[-1], seed=seed)
    frame_dict = {"train": train_frames, "val": val_frames, "test": test_frames}

    shared_args = (rec_id, output_dir, frame_dict, tracks_meta, tracks, lane_graph)

    return shared_args
