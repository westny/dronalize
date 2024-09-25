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
import sys
import time

import yaml
import pickle
import warnings
from typing import Any
from multiprocessing import Pool, Value, Lock

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.utils.urban_utils import preprocess_levelxd, preprocess_sind
from preprocessing.utils.common import *

worker_counter: Any
worker_lock: Any


def process_id(
        id0: int,
        rec_id: str,
        out_dir: str,
        fr_dict: dict,
        tr_meta: pd.DataFrame,
        tr: pd.DataFrame,
        ln_graph: dict,
        current_set: str = "train",
        dataset: str = "rounD",
        fz: int = 25,
        input_len: int = 3,
        output_len: int = 5,
        n_inputs: int = 7,
        n_outputs: int = 7,
        ds_factor: int = 5,
        filt_ord: int = 7,
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
    :param filt_ord: The filter order
    :param skip: The number of frames to skip
    :param dataset: The dataset name
    :param debug: Debug mode
    :return: None
    """

    not_set = get_other_sets(current_set)

    if not_set is None:
        not_set = ["val", "test"]
    df = tr[tr.trackId == id0]
    frames = df.frame.to_numpy()

    # Remove frames that are not in the current set
    frames = update_frames(frames, fr_dict[not_set[0]], fr_dict[not_set[1]])

    if len(frames) < fz * (input_len + output_len) + 1:
        return None
    for frame in frames[0:-1:skip]:  # Skip every 0.5 second
        prediction_frame = frame + fz * input_len
        final_frame = prediction_frame + fz * output_len
        if final_frame not in frames:
            break

        sas = get_neighbors(tr, prediction_frame - 1, id0)
        sa_ids = pd.unique(sas.trackId)
        n_sas = len(sa_ids)

        agent_ids = [id0, *sa_ids]

        # Retrieve meta information
        agent_type = class_list_to_int_list(get_meta_property(tr_meta, agent_ids, prop="class"))

        # Convert to tensors
        agent_type_tensor = torch.tensor(agent_type).long()

        input_array = np.empty((n_sas + 1, fz * input_len, n_inputs))
        target_array = np.empty((n_sas + 1, fz * output_len, n_outputs))

        for j, v_id in enumerate(agent_ids):
            input_array[j] = get_features(tr, frame, prediction_frame - 1, n_inputs, v_id)
            target_array[j] = get_features(tr, prediction_frame, final_frame - 1, n_outputs, v_id)

        # Down-sample the data
        if ds_factor > 1:
            input_array = decimate_nan(input_array, pad_order='front',
                                       ds_factor=ds_factor, fz=fz, filter_order=filt_ord)
            target_array = decimate_nan(target_array, pad_order='back',
                                        ds_factor=ds_factor, fz=fz, filter_order=filt_ord)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float()

        # Create masks
        three_sec = 3 * fz / ds_factor

        # Detect static_ids (we don't want to score on parked vehicles)
        non_scored_ids = []
        for j in range(len(agent_ids)):
            a = len(input_tensor[j, :, 0].unique())
            b = len(target_tensor[j, :, 0].unique())
            if a + b <= 3:
                non_scored_ids.append(j)
        if 0 in non_scored_ids:
            continue

        non_scored_ids_tensor = torch.tensor(non_scored_ids)

        input_mask, valid_mask, sa_mask, ma_mask = get_masks(
            input_tensor, target_tensor, int(three_sec), non_scored_ids_tensor
        )

        # make nans into zeros
        input_tensor[torch.isnan(input_tensor)] = 0.0
        target_tensor[torch.isnan(target_tensor)] = 0.0

        agent = {
            "num_nodes": n_sas + 1,
            "ta_index": 0,
            "ids": agent_ids,
            "type": agent_type_tensor,
            "inp_pos": input_tensor[..., :2],
            "inp_vel": input_tensor[..., 2:4],
            "inp_acc": input_tensor[..., 4:6],
            "inp_yaw": input_tensor[..., 6:],
            "trg_pos": target_tensor[..., :2],
            "trg_vel": target_tensor[..., 2:4],
            "trg_acc": target_tensor[..., 4:6],
            "trg_yaw": target_tensor[..., 6:],
            "input_mask": input_mask,
            "valid_mask": valid_mask,
            "sa_mask": sa_mask,
            "ma_mask": ma_mask,
        }

        data: dict[str, Any] = {'rec_id': rec_id, 'agent': agent}
        data.update(ln_graph)

        if not debug:
            with worker_lock:
                save_name = f"{dataset}_{current_set}_{worker_counter.value}"
                worker_counter.value += 1

            with open(f"{out_dir}/{current_set}/{save_name}.pkl", "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    return None


def process_ids(
        current_set: str,
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
    :param ln_graph: The lanelet graph
    """
    assert current_set in ["train", "val", "test"], "current_set must be one of [train, val, test]"

    fz = config["sample_freq"]
    ds = config["dataset"]
    input_len = config["input_len"]
    output_len = config["output_len"]
    n_inputs = config["n_inputs"]
    n_outputs = config["n_outputs"]
    ds_factor = config["downsample"]
    filt_ord = 2 if 'sind' in ds.lower() else 7
    skip = config["skip_samples"]
    debug = args.debug

    outer_args = (ds, fz, input_len, output_len, n_inputs, n_outputs, ds_factor, filt_ord, skip, debug)

    # Check if there are any saved samples in the current set directory
    set_dir = f"{output_dir}/{current_set}"
    if len(os.listdir(set_dir)) > 0:
        # get the highest save_id
        save_ids = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(set_dir)]
        save_id = max(save_ids) + 1
    else:
        save_id = 0

    save_id_counter = Value("i", save_id)
    save_lock = Lock()

    ta_ids = list(
        tr_meta[~tr_meta["class"].isin(["animal", "trailer"])].trackId
    )

    frame_range = fr_dict[current_set]
    ta_ids_set = {ta_id for ta_id in ta_ids if
                  any(tr[tr.trackId == ta_id].frame.isin(frame_range))}

    parked_vehicles = set(tr_meta[(tr_meta.initialFrame == 0) &
                                  (tr_meta.finalFrame == tr_meta.finalFrame.max())].trackId.values)

    ta_ids = list(ta_ids_set - parked_vehicles)

    arguments = [
        (ta_id, rec_id, out_dir, fr_dict, tr_meta,
         tr, ln_graph, current_set, *outer_args) for ta_id in ta_ids
    ]

    n_workers = 1
    if args.use_threads:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            warnings.warn("Could not determine the number of CPU cores. Using 1 thread.")
        elif cpu_count <= 2:
            warnings.warn("The number of CPU cores is too low. Using 1 thread.")
        else:
            n_workers = cpu_count

    with Pool(n_workers, initializer=init_worker,
              initargs=(save_id_counter, save_lock)) as pool:
        with tqdm(total=len(ta_ids), desc=f"{current_set.capitalize()}",
                  position=1, leave=False) as pbar:
            for _ in pool.imap_unordered(worker_function, arguments):
                pbar.update()


def init_worker(counter, lock):
    # Attach the counter and lock to the worker
    global worker_counter, worker_lock
    worker_counter, worker_lock = counter, lock


def worker_function(arg: tuple) -> None:
    # Wrapper function to call extract_by_frame with multiple arguments
    return process_id(*arg)


if __name__ == "__main__":
    if args.debug:
        print("DEBUG MODE: ON \n")

    config_file = args.config
    if not config_file.endswith(".yml"):
        config_file += ".yml"

    config_file_pth = os.path.join("preprocessing", "configs", config_file)

    assert os.path.exists(config_file_pth), f"Config file {config_file} not found."

    print(f"Using config file: {config_file} \n")

    with open(config_file_pth, "r", encoding="utf-8") as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = config["dataset"]

    output_dir = create_directories(args, dataset)
    print(f"Output directory: {output_dir} \n")

    random_seed = config["seed"]
    np.random.seed(random_seed)

    rec_ids = []
    recordings = config["recordings"]
    for key, value in recordings.items():
        if value["include"]:
            rec_ids.append(key)

    if dataset == "inD":
        temp_path = os.path.join(args.path, dataset, "maps", "lanelets")
        dirs = os.listdir(temp_path)

        # check if "01_bendplatz_constuction" is in the directory
        if "01_bendplatz_constuction" in dirs:
            # fix spelling error in directory
            os.rename(
                os.path.join(temp_path, "01_bendplatz_constuction"),
                os.path.join(temp_path, "01_bendplatz_construction"),
            )

    elif dataset == "uniD":
        temp_path = os.path.join(args.path, dataset, "maps")
        dirs = os.listdir(temp_path)

        # check if lanelet directory is named "lanelet" instead of "lanelets"
        if "lanelet" in dirs:
            # update name in directory for consistency
            os.rename(os.path.join(temp_path, "lanelet"), os.path.join(temp_path, "lanelets"))

    try:
        for r_id in tqdm(rec_ids, desc="Main process: ", position=0, leave=True):
            print(f"Preprocessing started for recording {r_id}...")

            if dataset.lower() in ("round", "ind", "unid"):
                shared_args = preprocess_levelxd(
                    args.path, r_id, config, output_dir, random_seed, dataset, args.debug
                )
            elif "sind" in dataset.lower():
                shared_args = preprocess_sind(
                    args.path, r_id, config, output_dir, random_seed, dataset, args.debug
                )
            else:
                raise ValueError(f"Dataset {dataset} not supported.")

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
