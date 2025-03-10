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
import time
import yaml
import pickle
import warnings
from typing import Any
from multiprocessing import Pool, Value, Lock

import pandas as pd
import numpy as np
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.utils import (
    # utils/highway_utils.py:
    preprocess_highd,
    preprocess_isac,
    # utils/exit_utils.py:
    preprocess_exid,
    # utils/common.py:
    create_tensor_dict,
    create_directories,
    erase_previous_line,
    get_other_sets,
    get_maneuver,
    get_meta_property,
    get_neighbors,
    get_features,
    update_frames,
    class_list_to_int_list

)

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
               filt_ord: int = 7,
               skip: int = 12,
               add_supp: bool = False,
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
    :param add_supp: Additional data
    :param debug: Debug mode
    :return: None
    """

    # Check if supplementary data should be added
    if add_supp:
        add_feats = ["laneDisplacement", "roadDisplacement"]
        n_inputs += 2
    else:
        add_feats = None

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
        intentions = get_maneuver(tr, prediction_frame - 1, agent_ids, prop='maneuver')
        agent_type = class_list_to_int_list(get_meta_property(tr_meta, agent_ids, prop='class'))

        input_array = np.empty((n_sas + 1, fz * input_len, n_inputs))
        target_array = np.empty((n_sas + 1, fz * output_len, n_outputs))

        for j, v_id in enumerate(agent_ids):
            input_array[j] = get_features(tr, frame, prediction_frame - 1, n_inputs, v_id, add_feats)
            target_array[j] = get_features(tr, prediction_frame, final_frame - 1, n_outputs, v_id)

        # Create the agent dictionary
        agent = create_tensor_dict(input_array, target_array,
                                   agent_ids, agent_type,
                                   fz, ds_factor, filt_ord,
                                   additional_features=add_feats,
                                   intentions=intentions)

        data: dict[str, Any] = {'rec_id': rec_id, 'agent': agent}
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
    filt_ord = 2 if ds.lower() == 'a43' else 7
    skip_lc = config["skip_lc_samples"]
    skip_lk = config["skip_lk_samples"]

    add_supp = args.add_supp
    debug = args.debug

    outer_lc_args = (ds, fz, input_len, output_len, n_inputs,
                     n_outputs, ds_factor, filt_ord, skip_lc, add_supp, debug)
    outer_lk_args = (ds, fz, input_len, output_len, n_inputs,
                     n_outputs, ds_factor, filt_ord, skip_lk, add_supp, debug)

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

    ta_ids = list(tr.trackId.unique())

    frame_range = fr_dict[current_set]
    ta_set = {ta_id for ta_id in ta_ids if any(tr[tr.trackId == ta_id].frame.isin(frame_range))}

    # Get the ids of all the TAs that perform lane changes
    lc_ta_ids = {ta_id for ta_id in ta_set if int(tr_meta[tr_meta.trackId == ta_id].numLaneChanges.iloc[0]) > 0}

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

    if dataset == "exiD":
        temp_path = os.path.join(args.path, dataset, "maps")
        dirs = os.listdir(temp_path)

        # check if lanelet directory is named "lanelet" instead of "lanelets"
        if "lanelet2" in dirs:
            # update name in directory for consistency
            os.rename(os.path.join(temp_path, "lanelet2"), os.path.join(temp_path, "lanelets"))

    try:
        for r_id in tqdm(rec_ids, desc="Main process: ", position=0, leave=True):
            print(f"Preprocessing started for recording {r_id}...")

            if dataset.lower() == "highd":
                shared_args = preprocess_highd(args.path, r_id, config, output_dir, random_seed,
                                               add_supp=args.add_supp, debug=args.debug)
            elif dataset.lower() == "exid":
                shared_args = preprocess_exid(args.path, r_id, config, output_dir, random_seed,
                                              add_supp=args.add_supp, debug=args.debug)
            elif dataset.lower() == "a43":
                shared_args = preprocess_isac(args.path, r_id, config, output_dir, random_seed,
                                              add_supp=args.add_supp, debug=args.debug)
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

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
