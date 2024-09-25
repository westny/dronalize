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
import warnings
from pathlib import Path

import torch
from torch.multiprocessing import set_sharing_strategy
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import Logger, CSVLogger

from arguments import args
from utils import load_config, import_from_module

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")

set_sharing_strategy('file_system')

# Load configuration and import modules
config = load_config(args.config)
TorchModel = import_from_module(config["model"]["module"], config["model"]["class"])
LitDataModule = import_from_module(config["datamodule"]["module"], config["datamodule"]["class"])
LitModel = import_from_module(config["litmodule"]["module"], config["litmodule"]["class"])


def main(save_name: str) -> None:
    ds = config["dataset"]
    path = Path("saved_models") / ds / save_name

    # Check if checkpoint exists
    if path.with_suffix(".ckpt").exists():
        ckpt = path.with_suffix(".ckpt")
    elif path.with_name(path.name + "-v1").with_suffix(".ckpt").exists():
        ckpt = path.with_name(path.name + "-v1").with_suffix(".ckpt")
    else:
        if not args.dry_run:
            raise NameError(f"Could not find model with name: {save_name}")

    # Determine the number of devices, and accelerator
    if torch.cuda.is_available() and args.use_cuda:
        devices, accelerator = -1, "auto"
    else:
        devices, accelerator = 1, "cpu"

    # Setup logger
    logger: bool | Logger
    if args.dry_run:
        logger = False
        args.small_ds = True
    elif not args.use_logger:
        logger = False
    else:
        logger = CSVLogger(save_dir=os.path.join("lightning_logs", ds), name=save_name)

    # Setup model
    net = TorchModel(config["model"])
    model = LitModel(net, config["training"])

    # Load checkpoint into model
    try:
        ckpt_dict = torch.load(ckpt, weights_only=True)
    except UnboundLocalError:
        if not args.dry_run:
            raise FileNotFoundError(f"Could not find checkpoint: {ckpt}")
    else:
        print(f"Loading checkpoint: {ckpt}")
        model.load_state_dict(ckpt_dict["state_dict"], strict=False)

    # Setup datamodule
    if args.root:
        config["datamodule"]["root"] = args.root
    datamodule = LitDataModule(config["datamodule"], args)

    # Setup trainer
    trainer = Trainer(accelerator=accelerator, devices=devices, logger=logger)

    # Start testing
    trainer.test(model, datamodule=datamodule, verbose=True)


if __name__ == "__main__":
    seed_everything(args.seed, workers=True)

    mdl_name = config["model"]["class"]
    ds_name = config["dataset"]
    add_name = f"-{args.add_name}" if args.add_name else ""

    full_save_name = f"{mdl_name}{add_name}-{ds_name}"

    print('----------------------------------------------------')
    print(f'\nGetting ready to test model: {full_save_name} \n')
    print('----------------------------------------------------')

    main(full_save_name)
