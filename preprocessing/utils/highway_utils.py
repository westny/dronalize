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

import numpy as np
from pandas import DataFrame, unique


def align_origin_w_centroid(tracks_meta: DataFrame,
                            tracks: DataFrame, debug: bool = False) -> DataFrame:
    """
    The coordinates are given wrt the upper left corner of the bounding box
    this function modifies the dataframe such that the coordinates are align
    with the center of the bounding box
    """
    if debug:
        return tracks

    ids = tracks_meta.trackId
    driving_dirs = tracks_meta.drivingDirection
    for i, dd in zip(ids, driving_dirs):
        tracks.loc[tracks.trackId == i, 'y'] += tracks.loc[tracks.trackId == i, 'height'] / 2
        if dd == 2:
            tracks.loc[tracks.trackId == i, 'x'] += tracks.loc[tracks.trackId == i, 'width'] / 2
    return tracks


def add_heading_feat(tracks: DataFrame, debug: bool = False) -> DataFrame:
    """
    Add heading as a feature to the tracks dataframe
    """
    tracks['psi'] = np.empty(len(tracks))

    if debug:
        return tracks

    t_ids = unique(tracks.trackId)
    for t_id in t_ids:
        vy = tracks.loc[tracks.trackId == t_id, 'vy'].to_numpy()
        vx = tracks.loc[tracks.trackId == t_id, 'vx'].to_numpy()
        psi = np.arctan2(vy, vx)
        tracks.loc[tracks['trackId'] == t_id, ['psi']] = psi
    return tracks


def add_maneuver(tracks_meta: DataFrame, tracks: DataFrame,
                 fz: int = 25, debug: bool = False) -> DataFrame:
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

    tracks['maneuver'] = np.ones(len(tracks), dtype=int) * 3

    if debug:
        return tracks

    t_ids = unique(tracks.trackId)
    for t_id in t_ids:
        if int(tracks_meta[tracks_meta.trackId == t_id].numLaneChanges.iloc[0]) > 0:
            df = tracks[tracks.trackId == t_id]
            dr_dir = df.drivingDirection.values[0]
            frames = df.frame.to_numpy()
            lanes = df.laneId.to_numpy()
            event_indices = [i for i in range(1, len(lanes)) if lanes[i] != lanes[i - 1]]

            for event_index in event_indices:
                five_seconds_prior = range(max(0, event_index - 5 * fz + 1), event_index)
                three_seconds_prior = range(max(0, event_index - 3 * fz + 1), event_index)
                one_second_prior = range(max(0, event_index - fz + 1), event_index)

                five_second_frames = (frames[i] for i in five_seconds_prior)
                three_second_frames = (frames[i] for i in three_seconds_prior)
                one_second_frames = (frames[i] for i in one_second_prior)

                delta_lane = lanes[event_index] - lanes[event_index - 1]
                if dr_dir == 1:
                    # Traveling from right to left. Lane index increases
                    maneuvers = lane_change_left if delta_lane > 0 else lane_change_right
                else:
                    # Traveling from left to right. Lane index increases
                    maneuvers = lane_change_right if delta_lane > 0 else lane_change_left

                tracks.loc[(tracks['trackId'] == t_id) &
                           (tracks['frame'].isin(five_second_frames)), ['maneuver']] = maneuvers[2]
                tracks.loc[(tracks['trackId'] == t_id) &
                           (tracks['frame'].isin(three_second_frames)), ['maneuver']] = maneuvers[1]
                tracks.loc[(tracks['trackId'] == t_id) &
                           (tracks['frame'].isin(one_second_frames)), ['maneuver']] = maneuvers[0]

    return tracks


def add_driving_direction(tracks_meta: DataFrame, tracks: DataFrame) -> DataFrame:
    """
    Add driving direction (1 or 2) as a feature to the tracks dataframe.
    If driving direction is 1, the vehicle is driving from right to left (negative x).
    If driving direction is 2, the vehicle is driving from left to right (positive x).
    """

    tracks['drivingDirection'] = np.empty(len(tracks))
    t_ids = unique(tracks.trackId)
    for t_id in t_ids:
        driving_direction = tracks_meta[tracks_meta.trackId == t_id].drivingDirection.values[0]
        tracks.loc[tracks['trackId'] == t_id, ['drivingDirection']] = driving_direction
    return tracks


def add_displacement_feat(rec_meta: DataFrame, tracks_meta: DataFrame,
                          tracks: DataFrame,
                          debug: bool = False) -> DataFrame:
    """
    Add roadDisplacement and laneDisplacement as features to the tracks dataframe.
    These features are used to determine the relative position
     of the vehicle with respect to the road and the lane.
    They could potentially replace the lane graph and be added
     to the input features of the model.
    """

    ulm = [float(l) for l in list(rec_meta['upperLaneMarkings'])[0].split(';')]
    llm = [float(l) for l in list(rec_meta['lowerLaneMarkings'])[0].split(';')]

    def compute_road_w():
        upper_l = ulm[-1] - ulm[0]
        lower_l = llm[-1] - llm[0]
        return upper_l, lower_l

    def compute_lane_w():
        upper_l = np.mean(np.diff(ulm))
        lower_l = np.mean(np.diff(llm))
        return np.mean([upper_l, lower_l])

    def get_road_edge_markings():
        return ulm[0], llm[0]

    def get_lane_markings():
        combined = ulm + llm
        return np.array(combined)

    def get_dyl(y, dd, lm, lw):
        dy = 2 * (y - lm) / lw - 1
        if dd == 2:
            dy *= (-1)
        return dy

    def get_dy(y, dd, curr_lane_id, lm, lw):
        dy = 2 * (y - lm[curr_lane_id - 2]) / lw - 1
        if dd == 2:
            dy *= (-1)
        return dy

    tracks['roadDisplacement'] = np.empty(len(tracks))
    tracks['laneDisplacement'] = np.empty(len(tracks))

    if debug:
        return tracks

    ur, lr = get_road_edge_markings()
    ruw, rlw = compute_road_w()

    lm = get_lane_markings()
    lw = compute_lane_w()
    t_ids = unique(tracks.trackId)
    for t_id in t_ids:
        driving_dir = int(tracks_meta[tracks_meta.trackId == t_id].drivingDirection.iloc[0])
        lane_ids = tracks.loc[tracks.trackId == t_id, 'laneId'].to_numpy()
        y = tracks.loc[tracks.trackId == t_id, 'y'].to_numpy()
        d_y = get_dy(y, driving_dir, lane_ids, lm, lw)

        marking, width = (ur, ruw) if driving_dir == 1 else (lr, rlw)
        d_y_r = get_dyl(y, driving_dir, marking, width)

        tracks.loc[tracks['trackId'] == t_id, ['laneDisplacement']] = d_y
        tracks.loc[tracks['trackId'] == t_id, ['roadDisplacement']] = d_y_r
    return tracks


def get_disp_features(df: DataFrame, frame_start: int, frame_end: int, track_id=-1) -> np.ndarray:
    return_array = np.empty((frame_end - frame_start + 1, 2))
    return_array[:] = np.NaN

    if track_id != -1:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end) & (df.trackId == track_id)]
    else:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]
    try:
        first_frame = dfx.frame.values[0]
    except IndexError:
        return return_array
    frame_offset = first_frame - frame_start

    features = dfx[['roadDisplacement', 'laneDisplacement']].to_numpy()

    return_array[frame_offset:frame_offset + features.shape[0], :] = features

    return return_array


def update_signs(rec_meta: DataFrame, tracks_meta: DataFrame,
                 tracks: DataFrame, debug: bool = False) -> DataFrame:
    """
    We are looking to unify the coordinate system under a
     FLU (frontward-leftward-upward) coordinate system (ISO standard):
    Forward motion = positive x
    Leftward motion = positive y
    (Upward motion = positive z)

    This requires updating the tracks differently depending
    on how the vehicles are moving (driving direction).
    To find the origin of the FLU coordinate system,
     we utilize the lower and upper lane markings.
    Longitudinal motion is updated based on the driving direction.

    """
    if debug:
        return tracks

    ulm = [float(x) for x in list(rec_meta['upperLaneMarkings'])[0].split(';')]
    llm = [float(x) for x in list(rec_meta['lowerLaneMarkings'])[0].split(';')]

    # subtract x_min from all tracks (to make everything start/end at 0)
    x_min = tracks.x.min()
    tracks.x -= x_min

    x_max = tracks.x.max()

    t_ids = unique(tracks.trackId)
    for t_id in t_ids:
        driving_dir = tracks_meta[tracks_meta.trackId == t_id].drivingDirection.values[0]

        if driving_dir == 1:
            tracks.loc[(tracks['trackId'] == t_id), ['y']] = \
                tracks.loc[(tracks['trackId'] == t_id), ['y']] - ulm[0]
            tracks.loc[(tracks['trackId'] == t_id), ['x']] = \
                -tracks.loc[(tracks['trackId'] == t_id), ['x']] + x_max
            tracks.loc[(tracks['trackId'] == t_id), ['vx']] = \
                -tracks.loc[(tracks['trackId'] == t_id), ['vx']]
            tracks.loc[(tracks['trackId'] == t_id), ['ax']] = \
                -tracks.loc[(tracks['trackId'] == t_id), ['ax']]
        else:
            tracks.loc[(tracks['trackId'] == t_id), ['y']] = \
                llm[-1] - tracks.loc[(tracks['trackId'] == t_id), ['y']]
            tracks.loc[(tracks['trackId'] == t_id), ['vy']] = \
                -tracks.loc[(tracks['trackId'] == t_id), ['vy']]
            tracks.loc[(tracks['trackId'] == t_id), ['ay']] = \
                -tracks.loc[(tracks['trackId'] == t_id), ['ay']]

    return tracks
