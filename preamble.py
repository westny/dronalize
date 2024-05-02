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


import json
import importlib
from typing import Callable
from pathlib import Path


def load_config(config: str) -> dict:
    # check if file contains ".json" extension
    if not config.endswith(".json"):
        config += ".json"

    # check if file exists in any of the config subdirectories
    config_path = Path("configs")

    # get all subdirectories
    subdirs = [d for d in config_path.iterdir() if d.is_dir()]

    # get all files in subdirectories
    files = [f for d in subdirs for f in d.iterdir() if f.is_file()]

    # check if config is any of the files
    if not any(config in f.name for f in files):
        raise FileNotFoundError(f"Config file {config} not found.")

    config = [str(f) for f in files if config in f.name][0]

    with open(config, 'r', encoding='utf-8') as openfile:
        conf = json.load(openfile)
    return conf


def import_module(module_name: str) -> object:
    return importlib.import_module(module_name)


def import_from_module(module_name: str, class_name: str) -> Callable:
    module = import_module(module_name)
    return getattr(module, class_name)
