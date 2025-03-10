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

from argparse import ArgumentParser, ArgumentTypeError


def str_to_bool(value: bool | str) -> bool:
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    true_vals = ("yes", "true", "t", "y", "1")
    false_vals = ("no", "false", "f", "n", "0")
    if isinstance(value, bool):
        return value
    if value.lower() in true_vals:
        return True
    if value.lower() in false_vals:
        return False
    raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser(description='Dronalize learning arguments')


# Program arguments
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--use-logger', type=str_to_bool, default=False,
                    const=True, nargs="?",
                    help='if logger should be used (default: False)')
parser.add_argument('--use-cuda', type=str_to_bool, default=False,
                    const=True, nargs="?",
                    help='if cuda exists and should be used (default: False)')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of workers in dataloader (default: 1)')
parser.add_argument('--pin-memory', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='if pin memory should be used (default: True)')
parser.add_argument('--persistent-workers', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='if persistent workers should be used (default: True)')
parser.add_argument('--store-model', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='if checkpoints should be stored (default: True)')
parser.add_argument('--overwrite', type=str_to_bool, default=False,
                    const=True, nargs="?",
                    help='overwrite if model exists (default: False)')
parser.add_argument('--add-name', type=str, default="",
                    help='additional string to add to save name')
parser.add_argument('--dry-run', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='verify the code and the model (default: True)')
parser.add_argument('--small-ds', type=str_to_bool, default=False,
                    const=True, nargs="?",
                    help='Use tiny versions of dataset for fast testing (default: False)')
parser.add_argument('--config', '-c', type=str, default="example",
                    help='config file path for experiment (default: example)')
parser.add_argument('--pre-train', '-pt', type=str, default="",
                    help='file containing a pre-trained model (default: none)')
parser.add_argument('--root', type=str, default="",
                    help='root path for dataset (default: "")')

args = parser.parse_args()
