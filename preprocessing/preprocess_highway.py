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

import math
import os
import sys
import time
import json
import pickle
import warnings
from typing import Any
from multiprocessing import Pool, Value, Lock

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.utils.highway_graph import get_highway_graph
from preprocessing.utils.highway_utils import *
from preprocessing.utils.common import *

worker_counter: Any
worker_lock: Any


def process_id(id0: int,
               rec_id: str,
               out_dir: str,
               fr_dict: dict,
               tr_meta: pd.DataFrame,
               tr: pd.DataFrame,
               ln_graph: dict,
               current_set: str = 'train',
               dataset: str = 'highD',
               fz: int = 25,
               input_len: int = 2,
               output_len: int = 5,
               n_inputs: int = 7,
               n_outputs: int = 7,
               ds_factor: int = 5,
               skip: int = 12,
               debug: bool = False,
               ) -> None:
    """
    Extracts the data for a given set of frames and saves it to a pickle file.
    :param id0: The trackId of the target vehicle
    :param rec_id: The ID of the recording
    :param out_dir: Output directory
    :param current_set: The current set (train, val, test)
    :param fr_dict: The frames to extract
    :param tr_meta: The meta-data of the tracks
    :param tr: The trajectory data
    :param ln_graph: The lane graph
    :param fz: The sampling frequency
    :param input_len: The length of the input sequence
    :param output_len: The length of the output sequence
    :param n_inputs: The number of input features
    :param n_outputs: The number of output features
    :param ds_factor: The down-sampling factor
    :param skip: The number of frames to skip
    :param dataset: The dataset name
    :param debug: Debug mode
    :return: None
    """

    not_set = get_other_sets(current_set)

    if not_set is None:
        not_set = ['val', 'test']
    df = tr[tr.trackId == id0]
    frames = df.frame.to_numpy()

    # Remove frames that are not in the current set
    frames = update_frames(frames, fr_dict[not_set[0]], fr_dict[not_set[1]])

    if len(frames) < fz * (input_len + output_len) + 1:
        return None

    driving_dir = int(tr_meta[tr_meta.trackId == id0].drivingDirection.iloc[0])

    # First, we filter out the frames where the target vehicle is performing a lane keep
    # that way we can sample more frames for lane changes
    lk_frames = []
    lc_frames = []
    for frame in frames[::skip]:
        prediction_frame = frame + fz * input_len
        final_frame = prediction_frame + fz * output_len
        if final_frame not in frames:
            break
        ta_intent = get_maneuver(tr, prediction_frame - 1, [id0], prop='maneuver')[0]
        if ta_intent == 3:
            lk_frames.append(frame)
        else:
            lc_frames.append(frame)

    n_lc = len(lc_frames)
    n_lk = len(lk_frames)

    # Our goal is to not sample more lane keep frames than lane change frames
    if n_lc > 0:
        keep_lk = min(n_lc, n_lk)
    else:
        keep_lk = min(n_lk, 5)

    # The stride is selected such that we retain 'keep_lk' lane keep frames
    k = max(math.ceil(n_lk / (keep_lk + 1)), 1)

    # 'Slicing' the lane keep frames assures that we sample
    # from all parts of the trajectory
    lk_frames = lk_frames[::k]

    # Combine the lane change and lane keep frames
    updated_frames = lc_frames + lk_frames

    for frame in updated_frames:
        prediction_frame = frame + fz * input_len
        final_frame = prediction_frame + fz * output_len

        sas = get_neighbors(tr, prediction_frame - 1, id0, driving_dir)
        sa_ids = pd.unique(sas.trackId)
        n_sas = len(sa_ids)

        agent_ids = [id0, *sa_ids]

        # Retrieve meta information
        intention = get_maneuver(tr, prediction_frame - 1, agent_ids, prop='maneuver')
        agent_type = class_list_to_int_list(get_meta_property(tr_meta, agent_ids, prop='class'))

        # Convert to tensors
        intention_tensor = torch.tensor(intention).long()
        agent_type_tensor = torch.tensor(agent_type).long()

        input_array = np.empty((n_sas + 1, fz * input_len, n_inputs))
        target_array = np.empty((n_sas + 1, fz * output_len, n_outputs))

        for j, v_id in enumerate(agent_ids):
            input_array[j] = get_features(tr, frame, prediction_frame - 1, n_inputs, v_id)
            target_array[j] = get_features(tr, prediction_frame, final_frame - 1, n_outputs, v_id)

        # Down-sample the data
        if ds_factor > 1:
            input_array = decimate_nan(input_array, pad_order='front', ds_factor=ds_factor, fz=fz)
            target_array = decimate_nan(target_array, pad_order='back', ds_factor=ds_factor, fz=fz)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float()

        # Create masks
        three_sec = 3 * fz / ds_factor

        input_mask, valid_mask, sa_mask, ma_mask = \
            get_masks(input_tensor, target_tensor, int(three_sec))

        # make nans into zeros
        input_tensor[torch.isnan(input_tensor)] = 0.
        target_tensor[torch.isnan(target_tensor)] = 0.

        agent = {'num_nodes': n_sas + 1,
                 'ta_index': 0,
                 'ids': agent_ids,
                 'type': agent_type_tensor,
                 'inp_pos': input_tensor[..., :2],
                 'inp_vel': input_tensor[..., 2:4],
                 'inp_acc': input_tensor[..., 4:6],
                 'inp_yaw': input_tensor[..., 6:],
                 'trg_pos': target_tensor[..., :2],
                 'trg_vel': target_tensor[..., 2:4],
                 'trg_acc': target_tensor[..., 4:6],
                 'trg_yaw': target_tensor[..., 6:],
                 'intention': intention_tensor,
                 'input_mask': input_mask,
                 'valid_mask': valid_mask,
                 'sa_mask': sa_mask,
                 'ma_mask': ma_mask}

        data: dict[str, Any] = {'rec_id': rec_id, 'agent': agent}
        # data['x_min'] = None
        # data['x_max'] = None
        data.update(ln_graph['upper_map'] if driving_dir == 1 else ln_graph['lower_map'])

        if not debug:
            with worker_lock:
                save_name = f"{dataset}_{current_set}_{worker_counter.value}"
                worker_counter.value += 1
            with open(f"{out_dir}/{current_set}/{save_name}.pkl", "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def process_ids(current_set: str,
                rec_id: str,
                out_dir: str,
                fr_dict: dict,
                tr_meta: pd.DataFrame,
                tr: pd.DataFrame,
                ln_graph: dict
                ) -> None:
    """
    Extracts the data for a given set of frames and saves it to a pickle file.
    :param current_set: The current set (train, val, test)
    :param rec_id: The recording ID
    :param out_dir: Output directory
    :param fr_dict: The frames to extract
    :param tr_meta: The meta-data of the tracks
    :param tr: The trajectory data
    :param ln_graph: The lane graph
    """
    assert current_set in ['train', 'val', 'test'], 'current_set must be one of [train, val, test]'

    fz = config["sample_freq"]
    ds = config["dataset"]
    input_len = config["input_len"]
    output_len = config["output_len"]
    n_inputs = config["n_inputs"]
    n_outputs = config["n_outputs"]
    ds_factor = config["downsample"]
    skip_lc = config["skip_lc_samples"]
    skip_lk = config["skip_lk_samples"]

    debug = args.debug

    outer_lc_args = (ds, fz, input_len, output_len, n_inputs,
                     n_outputs, ds_factor, skip_lc, debug)
    outer_lk_args = (ds, fz, input_len, output_len, n_inputs,
                     n_outputs, ds_factor, skip_lk, debug)

    # Check if there are any saved samples in the current set directory
    set_dir = f"{output_dir}/{current_set}"
    if len(os.listdir(set_dir)) > 0:
        # get the highest save_id
        save_ids = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(set_dir)]
        save_id = max(save_ids) + 1
    else:
        save_id = 0

    save_id_counter = Value('i', save_id)
    save_lock = Lock()

    ta_ids = list(tr_meta[tr_meta['class'].isin(['car', 'truck'])].trackId)
    frame_range = fr_dict[current_set]
    ta_set = {ta_id for ta_id in ta_ids
              if any(tr[tr.trackId == ta_id].frame.isin(frame_range))}

    # Get the ids of all the TAs that perform lane changes
    lc_ta_ids = {ta_id for ta_id in ta_set
                 if int(tr_meta[tr_meta.trackId == ta_id].numLaneChanges.iloc[0]) > 0}

    # Compute the ids of all the TAs that perform lane keeping
    lk_ta_ids = ta_set - lc_ta_ids

    frac = max(len(lc_ta_ids) // 10, 5)

    # Remove some of the lane keeping data
    lk_ta_ids = set(np.random.choice(list(lk_ta_ids), frac, replace=False))

    lc_arguments = [
        (ta_id, rec_id, out_dir, fr_dict, tr_meta, tr,
         ln_graph, current_set, *outer_lc_args) for ta_id in lc_ta_ids
    ]
    lk_arguments = [
        (ta_id, rec_id, out_dir, fr_dict, tr_meta, tr,
         ln_graph, current_set, *outer_lk_args) for ta_id in lk_ta_ids
    ]

    arguments = lc_arguments + lk_arguments

    if args.use_threads:
        n_workers = 1
        cpu_count = os.cpu_count()
        if cpu_count is None:
            warnings.warn("Could not determine the number of CPU cores. Using 1 thread.")
        elif cpu_count <= 2:
            warnings.warn("The number of CPU cores is too low. Using 1 thread.")
        else:
            n_workers = cpu_count

        with Pool(n_workers, initializer=init_worker,
                  initargs=(save_id_counter, save_lock)) as pool:
            with tqdm(total=len(arguments), desc=f"{current_set.capitalize()}",
                      position=1, leave=False) as pbar:
                for _ in pool.imap_unordered(worker_function, arguments):
                    pbar.update()
    else:
        for arg in tqdm(arguments, desc=f"{current_set.capitalize()}", position=1, leave=False):
            process_id(*arg)


def init_worker(counter, lock):
    # Attach the counter and lock to the worker
    global worker_counter, worker_lock
    worker_counter, worker_lock = counter, lock


def worker_function(arg: tuple) -> None:
    # Wrapper function to call extract_by_frame with multiple arguments
    return process_id(*arg)


def erase_previous_line(double_jump: bool = False):
    """Erase the previous line in the terminal."""
    sys.stdout.write('\x1b[1A')  # Move the cursor up one line
    sys.stdout.write('\x1b[2K')  # Clear the entire line
    if double_jump:
        sys.stdout.write('\x1b[1A')


if __name__ == "__main__":
    if args.debug:
        print("DEBUG MODE: ON \n")

    # worker_counter: Any
    # worker_lock: Any

    output_dir = create_directories(args)
    print(f"Output directory: {output_dir} \n")

    with open("preprocessing/configs/" + args.dataset + ".json",
              "r", encoding="utf-8") as conf_file:
        config = json.load(conf_file)

    random_seed = config["seed"]
    np.random.seed(random_seed)

    rec_ids = [f"{i:02}" for i in range(1, 60 + 1)]

    try:
        for r_id in tqdm(rec_ids, desc="Main process: ", position=0, leave=True):
            print(f"Preprocessing started for recording {r_id}...")

            # Construct the base directory path for your data
            base_dir = os.path.join(args.path, args.dataset, "data")

            # Use os.path.join for each specific file
            rec_meta_path = os.path.join(base_dir, f"{r_id}_recordingMeta.csv")
            tracks_meta_path = os.path.join(base_dir, f"{r_id}_tracksMeta.csv")
            tracks_path = os.path.join(base_dir, f"{r_id}_tracks.csv")

            # Read the CSV files
            rec_meta = pd.read_csv(rec_meta_path)
            tracks_meta = pd.read_csv(tracks_meta_path)
            tracks = pd.read_csv(tracks_path)

            # For the lanelet file, construct the path similarly
            upper_map, lower_map, x_min, x_max = \
                get_highway_graph(rec_meta, tracks,
                                  spacing=config["lane_graph"]["spacing"],
                                  buffer=config["lane_graph"]["buffer"])
            lane_graph = {'upper_map': upper_map, 'lower_map': lower_map}

            # Perform some initial renaming
            if 'trackId' not in tracks_meta.columns:
                tracks_meta.rename(columns={'id': 'trackId'}, inplace=True)
                tracks.rename(columns={'id': 'trackId'}, inplace=True)
            if 'vx' not in tracks.columns:
                tracks.rename(columns={'xVelocity': 'vx'}, inplace=True)
                tracks.rename(columns={'yVelocity': 'vy'}, inplace=True)
                tracks.rename(columns={'xAcceleration': 'ax'}, inplace=True)
                tracks.rename(columns={'yAcceleration': 'ay'}, inplace=True)
            if "x" not in tracks.columns:
                tracks.rename(columns={'xCenter': 'x'}, inplace=True)
                tracks.rename(columns={'yCenter': 'y'}, inplace=True)

            # Make class lowercase in tracks_meta
            tracks_meta['class'] = tracks_meta['class'].str.lower()

            tracks = align_origin_w_centroid(tracks_meta, tracks, debug=args.debug)
            tracks = add_driving_direction(tracks_meta, tracks)
            tracks = add_maneuver(tracks_meta, tracks, debug=args.debug)
            tracks = update_signs(rec_meta, tracks_meta, tracks, debug=args.debug)
            tracks = add_heading_feat(tracks, debug=args.debug)

            # Determine train, val, test split (by frames)
            train_frames, val_frames, test_frames = \
                get_frame_split(tracks_meta.finalFrame.array[-1],
                                seed=random_seed)
            frame_dict = {'train': train_frames, 'val': val_frames, 'test': test_frames}

            shared_args = (r_id, output_dir, frame_dict, tracks_meta, tracks, lane_graph)

            tasks = [
                ('train',) + shared_args,
                ('val',) + shared_args,
                ('test',) + shared_args
            ]

            # Erase preprocessing message
            erase_previous_line()

            # Print and immediately erase a "done" message (as an example)
            print("Preprocessing completed.")
            time.sleep(1)  # Just to let the user see the message
            erase_previous_line(True)

            for task in tasks:
                process_ids(*task)

    except KeyboardInterrupt:
        print("Interrupted.")

    finally:
        print("Finished.")
