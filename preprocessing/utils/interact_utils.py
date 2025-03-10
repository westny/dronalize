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
import pandas as pd


def find_interact_files(path):
    # get all files in the directory
    files = os.listdir(path)

    # remove files that are not .csv files
    files = [f for f in files if f.endswith(".csv")]

    # remove the file extension
    files_wo_ext = [f.split(".")[0] for f in files]

    # remove the final underscore and word after it
    files_wo_split = [f.rsplit("_", 1)[0] for f in files_wo_ext]

    return files_wo_split


def find_target_vehicle(data, case_id):
    """
    Find the target vehicle in a case based on longest duration/most frames
    """
    track_ids = data[data['case_id'] == case_id]['trackId'].unique()
    max_duration = 0
    ego_id = None

    for track_id in track_ids:
        track = data[(data['case_id'] == case_id) & (data['trackId'] == track_id)]
        duration = track['timestamp_ms'].max() - track['timestamp_ms'].min()
        if duration > max_duration:
            max_duration = duration
            ego_id = track_id

    return ego_id


def classify_ped_bike_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Efficiently classify agents marked as 'pedestrian/bicycle' into separate classes
    based on their maximum speed, considering case_id and track_id.
    """
    df = df.copy()

    # Mask for pedestrian/bicycle entries
    ped_bike_mask = df['agent_type'] == 'pedestrian/bicycle'

    if not ped_bike_mask.any():
        return df

    # Compute speeds in a single vectorized operation
    df['speed'] = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2)

    # Compute max speed for each (case_id, track_id) in one groupby operation
    max_speeds = df.groupby(['case_id', 'track_id'])['speed'].max()

    # Create a dictionary mapping (case_id, track_id) -> 'pedestrian' or 'bicycle'
    type_map = (max_speeds < 3.0).map({True: 'pedestrian', False: 'bicycle'})

    # Efficiently update agent_type using vectorized map
    df.loc[ped_bike_mask, 'agent_type'] = df.loc[ped_bike_mask, ['case_id', 'track_id']].apply(
        lambda row: type_map.get((row['case_id'], row['track_id']), 'pedestrian/bicycle'), axis=1
    )

    # Drop temporary 'speed' column
    df.drop(columns=['speed'], inplace=True)

    return df


def classify_ped_bike_robust(df):
    """
    Classify agents marked as 'pedestrian/bicycle' into separate classes based on
    their maximum speed, maximum acceleration, and turning behavior.
    Assumes `ax` and `ay` exist in `df`.
    """
    df = df.copy()

    # Mask for pedestrian/bicycle entries
    ped_bike_mask = df['agent_type'] == 'pedestrian/bicycle'

    if not ped_bike_mask.any():
        return df

    # Compute speed
    df['speed'] = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2)

    # Compute acceleration magnitude using existing ax, ay
    df['acceleration'] = np.sqrt(df['ax'] ** 2 + df['ay'] ** 2)

    # Compute turning rate (angular velocity)
    df['angular_velocity'] = (df['vx'] * df['ay'] - df['vy'] * df['ax']) / (df['speed'] + 1e-6)

    # Compute max values for each (case_id, track_id)
    stats = df.groupby(['case_id', 'track_id']).agg(
        max_speed=('speed', 'max'),
        max_acc=('acceleration', 'max'),
        max_turn_rate=('angular_velocity', 'max')
    )

    # Classification rules
    def classify(row):
        if row.max_speed > 3.0 or row.max_acc <= 3.0 or row.max_turn_rate <= 0.5:
            return 'bicycle'
        return 'pedestrian'

    # Apply classification
    stats['new_type'] = stats.apply(classify, axis=1)

    # Create a mapping dictionary
    type_map = stats['new_type']

    # Efficiently update agent_type using map
    df.loc[ped_bike_mask, 'agent_type'] = df.loc[ped_bike_mask, ['case_id', 'track_id']].apply(
        lambda row: type_map.get((row['case_id'], row['track_id']), 'pedestrian/bicycle'), axis=1
    )

    # Drop temporary columns
    df.drop(columns=['speed', 'acceleration', 'angular_velocity'], inplace=True)

    return df


def print_classification_stats(df):
    """Print the distribution of agent types after classification"""
    print("\nAgent type distribution:")
    print(df['agent_type'].value_counts())

    print("\nSpeed statistics for each type:")
    for agent_type in df['agent_type'].unique():
        mask = df['agent_type'] == agent_type
        speeds = np.sqrt(df[mask]['vx'] ** 2 + df[mask]['vy'] ** 2)
        print(f"\n{agent_type}:")
        print(f"  Max speed: {speeds.max():.2f} m/s")
        print(f"  Mean speed: {speeds.mean():.2f} m/s")
        print(f"  Median speed: {speeds.median():.2f} m/s")

        # Print number of unique tracks of this type
        n_tracks = df[mask][['case_id', 'track_id']].drop_duplicates().shape[0]
        print(f"  Number of unique tracks: {n_tracks}")


def compute_trajectory_derivatives(tracks, dt: float = 0.1):
    """Efficiently compute trajectory derivatives for all tracks."""
    tracks = tracks.copy()  # Avoid modifying input

    # Compute time differences per group (assuming 10Hz data)
    def compute_gradients(group):
        """Compute acceleration and yaw for a trajectory group."""
        # Only select required columns to avoid deprecation warning
        data = group[['vx', 'vy']]

        if len(data) < 2:  # Skip single-row groups
            return pd.DataFrame({
                'ax': np.zeros(len(data)),
                'ay': np.zeros(len(data)),
                'psi': np.zeros(len(data))
            }, index=data.index)

        # Use try-except to handle potential division by zero
        try:
            ax = np.gradient(data['vx'], dt, edge_order=1)
            ay = np.gradient(data['vy'], dt, edge_order=1)
            psi = np.arctan2(data['vy'], data['vx'])
        except:
            # Return NaN if computation fails
            return pd.DataFrame({
                'ax': np.zeros(len(data)),
                'ay': np.zeros(len(data)),
                'psi': np.zeros(len(data))
            }, index=data.index)

        return pd.DataFrame({'ax': ax, 'ay': ay, 'psi': psi}, index=data.index)

    # Apply function to each group efficiently, ensure index is reset
    results = (tracks.groupby(['case_id', 'track_id'])[['vx', 'vy']]
               .apply(compute_gradients)
               .reset_index(level=[0, 1], drop=True))  # Drop group indices

    # Update values directly using index alignment
    tracks['ax'] = results['ax']
    tracks['ay'] = results['ay']
    tracks['psi'] = results['psi']

    return tracks
