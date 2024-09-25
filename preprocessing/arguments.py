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


parser = ArgumentParser(description='Preprocessing arguments')

# Program arguments
parser.add_argument('--path', type=str, default="../datasets",
                    help='path to dataset')
parser.add_argument('--config', type=str, default="rounD",
                    help='name of config file (default: "rounD")')
parser.add_argument('--output-dir', type=str, default="data",
                    help='output directory for processed data')
parser.add_argument('--add-name', type=str, default="",
                    help='additional string to add to output-dir save name')
parser.add_argument('--use-threads', type=str_to_bool, default=False,
                    const=True, nargs="?", help='if multiprocessing should be used')
parser.add_argument('--debug', type=str_to_bool, default=False,
                    const=True, nargs="?", help='debugging mode')

args = parser.parse_args()
