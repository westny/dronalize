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

import sys
import os
import shutil
from typing import Optional
from argparse import Namespace

import torch
import numpy as np
from pandas import DataFrame
from scipy.signal import decimate
from sklearn.model_selection import train_test_split


def erase_previous_line(double_jump: bool = False):
    """Erase the previous line in the terminal."""
    sys.stdout.write("\x1b[1A")  # Move the cursor up one line
    sys.stdout.write("\x1b[2K")  # Clear the entire line
    if double_jump:
        sys.stdout.write("\x1b[1A")


def create_directories(args: Namespace, dataset: Optional[str] = None):
    """Create directories for processed data."""
    if dataset is None:
        data_dir = args.dataset + args.add_name
    else:
        data_dir = dataset + args.add_name

    output_dir = os.path.join(args.output_dir, data_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif args.debug:
        return output_dir
    else:
        print(f"Folder {output_dir} already exists.")
        inp = input("Would you like to overwrite it? (y/n) \n").lower()
        if inp == "y":
            print("Overwriting folder...")
            # Clear folder
            try:
                shutil.rmtree(output_dir)
            except FileNotFoundError as e:
                print(f"Failed to delete {output_dir}. Reason: {e}")
            else:
                os.makedirs(output_dir)
        else:
            print("Exiting...")
            sys.exit()

    # create subdirectories
    if not os.path.exists(output_dir + "/train"):
        os.makedirs(output_dir + "/train")
    if not os.path.exists(output_dir + "/val"):
        os.makedirs(output_dir + "/val")
    if not os.path.exists(output_dir + "/test"):
        os.makedirs(output_dir + "/test")

    return output_dir


def get_frame_split(n_frames: int, seed: int = 42, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the frames into train, validation, and test sets."""

    all_frames = list(range(1, n_frames + 1))

    # Divide all frames into ten lists of equal length
    frame_lists = np.array_split(all_frames, 10)

    # Split the lists into train (80%), validation (10%), and test (10%) sets
    train, valtest = train_test_split(frame_lists, test_size=test_size, random_state=seed)
    val, test = train_test_split(valtest, test_size=0.5, random_state=seed)

    # Sort the order of the arrays in the list and concatenate them
    train = np.concatenate(sorted(train, key=lambda x: x[0]))
    val = np.concatenate(sorted(val, key=lambda x: x[0]))
    test = np.concatenate(sorted(test, key=lambda x: x[0]))
    return train, val, test


def update_frames(agent_frames: np.ndarray,
                  alt_set1: np.ndarray, alt_set2: np.ndarray) -> np.ndarray:
    """
    Remove frames from the agent_frames that are in alt_set1 or alt_set2
    :param agent_frames: array of frames that are in the agent set
    :param alt_set1: array of frames that are in an alternative set
    :param alt_set2: array of frames that are in another alternative set
    :return:
    """
    assert agent_frames[-1] > agent_frames[0], "The frames are not in the correct order"
    idx_b = np.isin(agent_frames, alt_set1)
    idx_c = np.isin(agent_frames, alt_set2)
    idx = np.logical_or(idx_b, idx_c)
    return agent_frames[~idx]


def class_list_to_int_list(class_list: list[str]) -> list[int]:
    """
    Convert a list of class names to a list of integers
    :param class_list:
    :return:
    """
    class_to_int = {
        'car': 0,
        'van': 0,
        'trailer': 0,
        'truck': 1,
        'truck_bus': 2,
        'bus': 2,
        'motorcycle': 3,
        'bicycle': 4,
        'pedestrian': 5,
        'tricycle': 6,
        'animal': 7,
    }
    return [class_to_int[c] for c in class_list]


def get_other_sets(current_set: str) -> Optional[list[str]]:
    """
    Get the other sets (train, val, test) based on the current set.
    :param current_set: The current set
    :return: The other sets
    """
    match current_set:
        case 'train':
            return ['val', 'test']
        case 'val':
            return ['train', 'test']
        case 'test':
            return ['train', 'val']
        case _:
            return None


def get_neighbors(df: DataFrame, frame: int, id0: int,
                  driving_dir: Optional[int] = None) -> DataFrame:
    """
    Get the vehicles (except id0) present at a given frame.
    """
    if driving_dir is None:
        df1 = df[(df.frame == frame) & (df.trackId != id0)]
    else:
        df1 = df[(df.frame == frame) & (df.trackId != id0) & (df.drivingDirection == driving_dir)]
    return df1


def get_meta_property(tracks_meta: DataFrame, agent_ids: list, prop: str = 'class') -> list[str]:
    """
    Get a meta property of the agent from the tracks_meta DataFrame
    """
    prp = [tracks_meta[tracks_meta.trackId == v_id][prop].values[0] for v_id in agent_ids]
    return prp


def get_maneuver(tracks: DataFrame, frame: int,
                 agent_ids: list, prop='maneuver') -> list[int]:
    """
    Get the maneuver of the agents at a given frame
    """
    prp = [tracks[(tracks.trackId == v_id) & (tracks.frame == frame)][prop].values[0]
           for v_id in agent_ids]
    return prp


def get_features(df: DataFrame,
                 frame_start: int,
                 frame_end: int,
                 n_features: int,
                 track_id: int = -1) -> np.ndarray:
    """
    Get the features of the agent with id track_id in the frame range [frame_start, frame_end]
    """

    return_array = np.empty((frame_end - frame_start + 1, n_features))
    return_array[:] = np.nan

    if track_id != -1:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end) & (df.trackId == track_id)]
    else:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]
    try:
        first_frame = dfx.frame.values[0]
    except IndexError:
        return return_array
    frame_offset = first_frame - frame_start

    features = dfx[['x', 'y', 'vx', 'vy', 'ax', 'ay', 'psi']].to_numpy()

    return_array[frame_offset:frame_offset + features.shape[0], :] = features

    return return_array


def decimate_nan(x: np.ndarray,
                 pad_order: str = 'front',
                 ds_factor: int = 5,
                 fz: int = 25,
                 max_s: float = 1.0,
                 filter_order: int = 7) -> np.ndarray:
    decimation_bound = max_s * fz
    target_ds_len = int(x.shape[1] / ds_factor)
    y = np.zeros((x.shape[0], target_ds_len, x.shape[2]))

    not_nan_idx = ~np.isnan(x[..., 0])
    decimation_check = not_nan_idx.sum(axis=1) >= decimation_bound
    decimation_idx = np.where(decimation_check)[0]
    lazy_idx = np.where(~decimation_check)[0]

    for idx in decimation_idx:
        arr = x[idx]

        # slice out the non-nan values
        arr = arr[not_nan_idx[idx]][::-1]

        # decimate the array
        arr = decimate(arr, ds_factor, n=filter_order, axis=0, zero_phase=True)[::-1]

        # pad the array with NaNs to the target length along the first axis
        if pad_order == 'front':
            n_pad = ((target_ds_len - arr.shape[0], 0), (0, 0))
            arr = np.pad(arr, n_pad, mode='constant', constant_values=np.nan)
        elif pad_order == 'back':
            n_pad = ((0, target_ds_len - arr.shape[0]), (0, 0))
            arr = np.pad(arr, n_pad, mode='constant', constant_values=np.nan)
        else:
            raise ValueError("pad_order should be either 'front' or 'back'")

        y[idx] = arr

    for idx in lazy_idx:
        y[idx] = x[idx, -1:0:-ds_factor][::-1]

    return y


def get_masks(x: torch.Tensor,
              y: torch.Tensor,
              ma_frames: int = 15,
              non_scored_ids: Optional[torch.Tensor] = None,
              k_max: int = 8) -> list[torch.Tensor]:
    """
    Get masks for the input and output tensors.
    There are four different masks that are created:

    1. input_mask: Mask for the input tensor (contains True for valid values (not NaNs))

    2. valid_mask: Mask for the output tensor (contains True for valid values (not NaNs)).
    This mask is recommended to be used for the loss calculation during training.

    3. ta_mask: Mask for the target agent (contains True for the target agent).
    This mask is used to identify the target agent in the single-target task.
    It is recommended to be used for quantitative metrics calculation for validation and testing.

    4. ma_mask: Mask for the multi-agent task (contains True for the surrounding agents).
    This mask is used to identify the surrounding agents in the multi-target task.
    It is recommended to be used for quantitative metrics calculation for validation and testing.
    For the multi-target task, the 8 closest agents to the target agent are selected.
    The requirement is that the agent should be visible for at least 3 seconds in to the future.

    :param x: The input tensor
    :param y: The output tensor
    :param ma_frames: The number of frames for the multi-agent mask (should represent 3 seconds)
    :param non_scored_ids: The ids of the agents that should not be scored
    :param k_max: The maximum number of agents to consider for the multi-agent mask
    :return: input_mask, valid_mask, tv_mask, mv_mask
    """
    input_mask = torch.isnan(x).sum(dim=2) == 0
    valid_mask = torch.isnan(y).sum(dim=2) == 0

    # Target agent mask. This will always have adequate number of samples by design
    ta_mask = torch.zeros_like(valid_mask)
    ta_mask[0] = True

    # Multi-agent mask

    # Only interested in indices where valid mask is True for at least 3 s
    long_idx = torch.where(valid_mask.sum(dim=1) >= ma_frames)[0]

    # Remove non-scored agents from long_idx (e.g., parked vehicles)
    if non_scored_ids is not None:
        long_idx = long_idx[~torch.isin(long_idx, non_scored_ids)]

    # (find the (max) 8 closest agents)
    dist = torch.norm(x[long_idx, -1, :2] - x[0, -1, :2], dim=1)
    max_k = min(k_max, len(long_idx))  # topk fails if k > input_tensor.shape[0]
    _, indices = torch.topk(dist, max_k, largest=False)

    intersection = long_idx[indices]

    ma_mask = torch.zeros_like(valid_mask)
    ma_mask[intersection] = True
    ma_mask = ma_mask & valid_mask  # only keep indices where valid mask is True

    return [input_mask, valid_mask, ta_mask, ma_mask]
