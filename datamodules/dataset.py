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
import pickle
import warnings
from typing import Optional, Callable
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DroneDataset(Dataset):
    def __init__(self,
                 root: str,
                 dataset: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 small_data: bool = False) -> None:
        super().__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)
        assert split in ['train', 'val', 'test'], 'Split must be one of [train, val, test]'

        self.root = root
        self.dataset = dataset
        self.split = split
        self.path = os.path.join(self.root, self.dataset, self.split)
        self.files = os.listdir(self.path)
        self.files = sorted(self.files)  # sort files for consistency across operating systems

        if small_data:
            self.files = self.files[:100]

        self._num_samples = len(self.files)

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        with open(os.path.join(self.path, self.files[idx]), 'rb') as f:
            return HeteroData(pickle.load(f))


if __name__ == "__main__":
    ds = DroneDataset(root='data', dataset='highD', split='test')
    print(len(ds))
