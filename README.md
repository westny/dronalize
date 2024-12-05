
<div align="center">
<img alt="Dronalize logo" src=https://github.com/westny/dronalize/assets/60364134/862a8a60-4cd0-4b21-b0d2-a4ee0e5b4f03 width="800px" style="max-width: 100%;">

______________________________________________________________________

[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=Dronalize&color=8A2BE2&logo=arxiv)](https://arxiv.org/abs/2405.00604)
[![python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.2-blue.svg)](https://pytorch.org/)
[![contributions](https://img.shields.io/badge/Contributions-welcome-297D1E)](#contributing)
[![license](https://img.shields.io/badge/License-Apache%202.0-2F2F2F.svg)](LICENSE)
<br>
[![Docker Status](https://github.com/westny/dronalize/actions/workflows/docker-image.yml/badge.svg)](.github/workflows/docker-image.yml)
[![Apptainer Status](https://github.com/westny/dronalize/actions/workflows/apptainer-image.yml/badge.svg)](.github/workflows/apptainer-image.yml)
[![Conda Status](https://github.com/westny/dronalize/actions/workflows/conda.yml/badge.svg)](.github/workflows/conda.yml)
[![Linting Status](https://github.com/westny/dronalize/actions/workflows/mypy.yml/badge.svg)](.github/workflows/mypy.yml)

</div>

**Dronalize** is a toolbox designed to alleviate the development efforts of researchers working with various drone datasets on behavior prediction problems.
It includes tools for data preprocessing, visualization, and evaluation, as well as a model development pipeline for data-driven motion forecasting.
<br> The toolbox utilizes [<img alt="Pytorch logo" src=https://github.com/westny/dronalize/assets/60364134/b6d458a5-0130-4f93-96df-71374c2de579 height="12">PyTorch](https://pytorch.org/docs/stable/index.html), [<img alt="PyG logo" src=https://github.com/westny/dronalize/assets/60364134/53554175-0ca1-4020-b8eb-7bbd4ebe0e47 height="12">PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), and [<img alt="Lightning logo" src=https://github.com/westny/dronalize/assets/60364134/167a7cbb-8346-44ac-9428-5f963ada54c2 height="16">PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for its functionality.

**📰 Latest Updates**

- 🚀 Added 4 new datasets to the toolbox: *exiD*, *uniD*, *SIND*, and *A43*.
- 🔧 Added more attributes to the lane graphs and improved preprocessing scripts.
- 📦 Added pre-built Docker image to Docker Hub.
- 🐍 Added PyPi installation instructions.

***

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Related Work](#related-work)
- [Contributing](#contributing)
- [Cite](#cite)

# Installation
There are several alternatives to installation, depending on your needs and preferences.
Our recommendation and personal preference is to use containers for reproducibility and consistency across different environments.
We have provided both an Apptainer `.def` file and a `Dockerfile` for this purpose.
Both recipes use the `mamba` package manager for creating the environments. 
In both cases, they utilize the same `environment.yml` file that could also be used to create a local conda environment if desired.
Additionally, we provide a `requirements.txt` file for those who prefer to use `pip` for package management.
All necessary files to install the required dependencies are found in the [build](build) directory.

### <img alt="Apptainer logo" src=https://github.com/westny/dronalize/assets/60364134/6a9e51ae-c6ce-4ad1-b79f-05ca7d959062 width="110">
[Apptainer](https://apptainer.org/docs/user/main/index.html) is commonly used in high-performance computing (HPC) for creating secure, portable, and reproducible environments. It is well-suited for research and scientific workflows.
It is a lightweight containerization tool that we prefer for its simplicity and ease of use.

<details>
  <summary>Click here for Installation Instructions</summary>

### Installation Instructions:

If you have not already done so, start by installing Apptainer on your system by following the instructions on the [Apptainer website](https://apptainer.org/docs/user/main/quick_start.html#installation).

#### Option 1: Pull a Pre-built Image from Docker Hub

You can pull a pre-built image from Docker Hub by running the following command:

```bash
apptainer pull dronalize.sif docker://westny/dronalize:latest
```

This will download the latest version of the image to your local machine.


#### Option 2: Build the Image Locally
You can build the container by running the following command:

```bash
apptainer build dronalize.sif /path/to/definition_file
```

where `/path/to/definition_file` is the path to the `apptainer.def` file in the repository.

### Running the Container
Once built, it is very easy to run the container as it only requires a few extra arguments. 
For example, to start the container and execute the `train.py` script, you can run the following command from the repository root directory:

```bash
apptainer run /path/to/dronalize.sif python train.py
```

If you have CUDA installed and want to use GPU acceleration, you can add the `--nv` flag to the `run` command.

```bash
apptainer run --nv /path/to/dronalize.sif python train.py
```

</details>

### <img alt="Docker logo" src=https://github.com/westny/dronalize/assets/60364134/1bf2df76-ab44-4bae-9623-03710eff0572 width="100">
[Docker](https://www.docker.com/get-started/) is a widely adopted platform for automating the deployment and management of containerized applications. It is suitable for users familiar with containers or those needing an isolated runtime environment.

<details>
  <summary>Click here for Installation Instructions</summary>

  ### Installation Instructions:

If you have not already done so, start by installing Docker on your system by following the instructions on the [Docker website](https://docs.docker.com/get-docker/).

#### Option 1: Pull a Pre-built Image from Docker Hub

You can pull a pre-built image from Docker Hub by running the following command:

```bash
docker pull westny/dronalize:latest
```

This will download the latest version of the image to your local machine.

We recommend tagging the image for easier use:

```bash
docker tag westny/dronalize:latest dronalize
```

#### Option 2: Build the Image Locally

You can build the image by running the following command from the container root directory:

```bash 
docker build -t dronalize .
```

This will create a Docker image named `dronalize` with all the necessary dependencies.


### Running the Container

To run the container, you can use the following command:

```bash
docker run -it dronalize
```

Note that training using docker requires mounting the data directory to the container.
Example of how this is done from the repository root directory:

```bash
docker run -v "$(pwd)":/app -w /app dronalize python train.py
```

### GPU Acceleration

To use GPU acceleration, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Add the NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
and run the container with the `--gpus all` flag.

```bash
docker run --gpus all -v "$(pwd)":/app -w /app dronalize python train.py
```

</details>

### <img alt="Conda logo" src=https://github.com/westny/dronalize/assets/60364134/52d02aa9-6231-4261-8e0f-6c092991c89c width="100">
[Conda](https://conda.io/projects/conda/en/latest/index.html) is a package and environment manager that allows users to create isolated environments without using containers.
It is useful for managing dependencies in Python and other languages.

<details>
  <summary>Click here for Installation Instructions</summary>

  ### Installation Instructions:

You can create a [conda](https://conda.io/projects/conda/en/latest/index.html) environment using the provided `environment.yml` file.

To create the environment, run the following command:

```bash
conda env create -f /path/to/environment.yml
```
or if using [mamba](https://mamba.readthedocs.io/en/latest/)
    
```bash
mamba env create -f /path/to/environment.yml
```

This will create a new conda environment named `dronalize` with all the necessary dependencies.
Once the environment is created, you can activate it by running:

```bash
conda activate dronalize
```

The environment is now ready to use, and you can run the scripts in the repository.

To deactivate the environment, run:

```bash
conda deactivate
```

</details>

### <img alt="Pypi logo" src=https://github.com/user-attachments/assets/41e5853c-35db-4b00-8b35-c888a1b55979 width="100">
<a id="pypi"></a>
Using `pip` to install dependencies directly from PyPI is a straightforward approach. This option works well for users who prefer not to use containers or conda environments but want to manage dependencies via a `requirements.txt` file.
We recommend using a virtual environment to avoid conflicts with other packages.

<details>
  <summary>Click here for Installation Instructions</summary>

  ### Installation Instructions:

First, create a new virtual environment using `venv`:

```bash
python3.x -m venv dronalize
```
where `x` is the version of Python you are using, e.g., `3.11` (used in the containers).

Activate the virtual environment:
```bash
source dronalize/bin/activate
```

Then install the required packages using `pip`:
```bash
pip install -r /path/to/requirements.txt
```

The environment is now ready to use, and you can run the scripts in the repository.

To deactivate the virtual environment, run:

```bash
deactivate
```

Anytime you want to use the environment, you need to activate it again.
</details>

<br>

# Usage
The **Dronalize** toolbox is designed for two main purposes: data preprocessing and evaluation of trajectory prediction models.
It was developed to be used in conjunction with [<img alt="PyTorch logo" src=https://github.com/westny/dronalize/assets/60364134/b6d458a5-0130-4f93-96df-71374c2de579 height="12">PyTorch](https://pytorch.org/docs/stable/index.html); in particular, the [<img alt="Lightning logo" src=https://github.com/westny/dronalize/assets/60364134/167a7cbb-8346-44ac-9428-5f963ada54c2 height="16">PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)  framework.

### Preprocessing

> Before running the preprocessing scripts, make sure to unzip the downloaded datasets and place them in a directory of your choice.
> By default, the scripts expect the datasets to be located in the `../datasets` directory, but this can be changed by specifying the `--path` argument.
> Make sure the unzipped folders are named `highD`, `rounD`, `inD`, `exiD`, `uniD`, `SIND`*, and `A43` respectively.
> 
> *Please see the [Datasets](#datasets) section for additional information on `SIND` file structure.

All functionality related to data preprocessing is contained in the `preprocessing` module.
Since the datasets have minor differences in their structure, there are separate scripts for preprocessing depending on the dataset used.
For example, to preprocess the `inD`, `rounD`, `uniD` or `SIND` datasets, you can run the following command (replace `dataset_name` with the respective dataset name):

```bash
python -m preprocessing.preprocess_urban.py --config 'dataset_name' --path 'path/to/datasets'
```

for the `highD`, `exiD` or `A43` datasets, you should run:

```bash
python -m preprocessing.preprocess_highway.py --config 'dataset_name' --path 'path/to/datasets'
```

By default, these scripts will save the preprocessed data in the `data` directory, this can be changed by specifying the `--output-dir` argument.
There is an option to use threading for faster processing by setting the `--use-threads` flag that we recommend for efficient processing.

There are additional **default** arguments in the respective configuration files within the `preprocessing/config` directory that should not be changed to facilitate consistency across different studies.
Finally, `preprocess.sh` is a script that can be used to preprocess all datasets in one go using the default arguments that we recommend for consistency.

```bash
. preprocess.sh
```

Using Apptainer, the shell script can be executed as follows:
```bash
  apptainer run /path/to/dronalize.sif bash preprocess.sh
``` 

> Depending on the dataset, the number of workers, and your hardware, preprocessing can some time.
> Expect a few hours to process **all** datasets with threading enabled. Of course, this only needs to be done once.


### Data Loading
In [datamodules](datamodules), you will find the necessary classes for loading the preprocessed data into PyTorch training pipelines.
It includes:
- `DroneDataset`: A `Dataset` class built around `torch_geometric`. Found in: [dataset.py](datamodules/dataset.py)
- `DroneDataModule`: A `DataModule` class, including `Dataloader` built around `lightning.pytorch`. Found in: [dataloader.py](datamodules/dataloader.py)
- `CoordinateTransform` and `CoordinateShift`: Example transformations. Found in: [transforms.py](datamodules/transforms.py)

> [dataloader.py](datamodules/dataloader.py) is designed to be runnable as a standalone script for quick testing of the data loading pipeline.
It includes a `main` function that can be used to load the data and visualize it for debugging and/or educational purposes.


### Modeling
In [models/prototype.py](models/prototype.py), there is a baseline neural network for trajectory prediction.
It is a simple encoder-decoder model that takes as input the past trajectory of a road user and outputs a predicted future trajectory.
It learns interactions between road users by encoding the scene as a graph and uses a GNN to process the data. 
The model could be used as a starting point for developing more advanced models, where adding map-aware mechanisms would be a natural next step.

```python
# prototype.py
import torch
import torch.nn as nn
import torch_geometric.nn as ptg
from torch_geometric.data import HeteroData


class Net(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_inputs = config["num_inputs"]
        num_outputs = config["num_outputs"]
        num_hidden = config["num_hidden"]
        self.ph = config["pred_hrz"]

        self.embed = nn.Linear(num_inputs, num_hidden)
        self.encoder = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.interaction = ptg.GATv2Conv(num_hidden, num_hidden, concat=False)
        self.decoder = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.output = nn.Linear(num_hidden, num_outputs)

    def forward(self, data: HeteroData) -> torch.Tensor:
        edge_index = data['agent']['edge_index']
        x = torch.cat([data['agent']['inp_pos'],
                       data['agent']['inp_vel'],
                       data['agent']['inp_yaw']], dim=-1)

        x = self.embed(x)
        _, h = self.encoder(x)
        x = h[-1]

        x = self.interaction(x, edge_index)
        x = x.unsqueeze(1).repeat(1, self.ph, 1)
        x, _ = self.decoder(x)

        pred = self.output(x)

        return pred
```

### Model Training
The toolbox includes a training script, [train.py](train.py), that can be used to train your models on the preprocessed data.
The script is designed to be run from the repository root directory and includes several arguments that can be used to configure the training process.
By default, it uses configuration files in `.json` format found in the [configs](configs) directory, detailing the required modules and hyperparameters for training.
Additional runtime arguments, such as the number of workers, GPU acceleration, debug mode, and model checkpointing, can be specified when running the script (see [arguments.py](arguments.py) for more information).

The training script is designed to be used with PyTorch Lightning; besides using the custom data modules previously mentioned, it also requires a `LightningModule` that defines the model and training loop.
In [base.py](base.py), you will find a base class that can be modified to build your own `LightningModule`. 
In its current form, it can be used to train and evaluate the baseline model.
It also details how to use the proposed evaluation metrics for trajectory prediction.

An example of how to train the model is shown below:
```bash
  [apptainer run --nv path/to/dronalize.sif] python train.py --add-name test --dry-run 0 --use-cuda 1 --num-workers 4
```

We recommend users modify the default arguments in [arguments.py](arguments.py) to suit their needs.

> Note that the default logger is set to `wandb` ([weights & biases](https://wandb.ai/)) for logging performance metrics during training.
> It is our preferred tool for tracking experiments, but it can be easily replaced with other logging tools by modifying the `Trainer` in the training script.
> 
> See the official [Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for more information on customizing training behavior and how to use the library in general.

### Metrics
The toolbox includes several evaluation metrics for trajectory prediction, implemented in the [`metrics`](metrics) module.
The metrics are designed to handle both uni- and multi-modal predictions.
Predictions are expected to be in the form of `(batch_size, num_timesteps, 2)` or `(batch_size, num_timesteps, num_modes, 2)`, where `num_modes` is the number of modes in the prediction.
> There is also support for mode-first predictions of shape `(batch_size, num_modes, num_timesteps, 2)` that can be used by setting the `mode_first` flag to `True`.
> Users can of course change the default behavior by directly modifying the metrics. 

Most metrics are also compatible with specifying a `min_criterion` (`FDE`, `ADE`, `MAP`) that is used to select which of the modes to evaluate against the ground-truth target (Default: `FDE`).
Setting `min_criterion` to `MAP` will evaluate the metrics based on the mode with the highest predicted probability. 
Note that `MAP` can only be used in conjunction with the optional argument `Prob` of shape `(batch_size, num_modes)` representing the weights of each mode.

The following metrics are implemented:
- [**Min. Average Displacement Error (minADE)**](metrics/min_ade.py)
- [**Min. Final Displacement Error (minFDE)**](metrics/min_fde.py)
- [**Min. Average Path Displacement Error (minAPDE)**](metrics/min_apde.py)
- [**Miss Rate**](metrics/miss_rate.py)
- [**Collision Rate**](metrics/collision_rate.py)
- [**Min. Brier**](metrics/min_brier.py)
- [**Negative Log-Likelihood (NLL)**](metrics/log_likelihood.py)

For their mathematical definitions, please refer to the paper.

<br>

# Datasets

The toolbox has been developed for use of the *[highD](https://levelxdata.com/highd-dataset/)*, *[rounD](https://levelxdata.com/round-dataset/)*, *[inD](https://levelxdata.com/ind-dataset/)*, *[exiD](https://levelxdata.com/exid-dataset/)*, *[uniD](https://levelxdata.com/unid-dataset/)*, *[SIND](https://github.com/SOTIF-AVLab/SinD)*, and [A43](https://data.isac.rwth-aachen.de/?p=58) datasets.
The datasets contain recorded trajectories from different locations in Germany and China, including various highways, roundabouts, and intersections.
Their high quality make them particularly suitable for early-stage research and development.
They are freely available for non-commercial use, which is our targeted audience, but most require applying for usage through the links: 

<div align="center">

| Dataset | Link                                     | Notes                                   |
|---------|------------------------------------------|-----------------------------------------|
| *highD* | https://levelxdata.com/highd-dataset/    |                                         |
| *rounD* | https://levelxdata.com/round-dataset/    |                                         |
| *inD*   | https://levelxdata.com/ind-dataset/      |                                         |
| *exiD*  | https://levelxdata.com/exid-dataset/     |                                         |
| *uniD*  | https://levelxdata.com/unid-dataset/     |                                         |
| *SIND*  | https://github.com/SOTIF-AVLab/SinD      | Visit the GitHub link for email request |
| *A43*   | https://data.isac.rwth-aachen.de/?p=58   | Directly downloadable at the link       |


</div>

> Several datasets in the leveLXData suite were recently updated (April 2024) that include improvements to the maps, as well as the addition of some new locations.
> This toolbox is designed to work with the updated datasets, and we recommend using the latest versions for the most recent features to avoid having to modify the toolbox.
***

### *[highD](https://arxiv.org/abs/1810.05642)*: The Highway Drone Dataset

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Scenario-based testing for the safety validation of
        highly automated vehicles is a promising approach that is being
        examined in research and industry. This approach heavily relies
        on data from real-world scenarios to derive the necessary
        scenario information for testing. Measurement data should be
        collected at a reasonable effort, contain naturalistic behavior of
        road users and include all data relevant for a description of the
        identified scenarios in sufficient quality. However, the current
        measurement methods fail to meet at least one of the
        requirements. Thus, we propose a novel method to measure data
        from an aerial perspective for scenario-based validation
        fulfilling the mentioned requirements. Furthermore, we provide
        a large-scale naturalistic vehicle trajectory dataset from German
        highways called highD. We evaluate the data in terms of
        quantity, variety and contained scenarios. Our dataset consists
        of 16.5 hours of measurements from six locations with 110 000
        vehicles, a total driven distance of 45 000 km and 5600 recorded
        complete lane changes.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @inproceedings{highDdataset,
       title={The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems},
       author={Krajewski, Robert and Bock, Julian and Kloeker, Laurent and Eckstein, Lutz},
       booktitle={2018 21st International Conference on Intelligent Transportation Systems (ITSC)},
       pages={2118-2125},
       year={2018},
       doi={10.1109/ITSC.2018.8569552}
    }
</details>

> #### Dataset Overview
> - Naturalistic trajectory dataset on six different recording locations
> - In total ~ 110,500 vehicles
> - Road user classes: car, trucks

<div align="center">
  <img src=https://github.com/westny/dronalize/assets/60364134/0e9de880-9ee3-4941-ab41-692f259a0cbc alt="highD.gif">
</div>

***

### *[rounD](https://ieeexplore.ieee.org/document/9294728)*: The Roundabouts Drone Dataset

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        The development and validation of automated vehicles involves a large number of challenges to be overcome.
        Due to the high complexity, many classic approaches quickly reach their limits and data-driven methods become necessary.
        This creates an unavoidable need for trajectory datasets of road users in all relevant traffic scenarios.
        As these trajectories should include naturalistic and diverse behavior, they have to be recorded in public traffic.
        Roundabouts are particularly interesting because of the density of interaction between road users, which must be considered by an automated vehicle for behavior planning.
        We present a new dataset of road user trajectories at roundabouts in Germany.
        Using a camera-equipped drone, traffic at a total of three different roundabouts in Germany was recorded.
        The tracks consisting of positions, headings, speeds, accelerations and classes of objects were extracted from recorded videos using deep neural networks.
        The dataset contains a total of six hours of recordings with more than 13 746 road users including cars, vans, trucks, buses, pedestrians, bicycles and motorcycles.
        In order to make the dataset as accessible as possible for tasks like scenario classification, road user behavior prediction or driver modeling, we provide source code for parsing and visualizing the dataset as well as maps of the recording sites.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @inproceedings{rounDdataset,
        title={The rounD Dataset: A Drone Dataset of Road User Trajectories at Roundabouts in Germany},
        author={Krajewski, Robert and Moers, Tobias and Bock, Julian and Vater, Lennart and Eckstein, Lutz},
        booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)},
        pages={1-6},
        year={2020},
        doi={10.1109/ITSC45102.2020.9294728}
    }
</details>

> #### Dataset Overview
> - Naturalistic trajectory dataset on three different recording locations
> - In total ~ 13,740 road users
> - Road user classes: car, trailer, truck, bus, motorcycle, bicycle, pedestrian

<div align="center">
  <img src=https://github.com/westny/dronalize/assets/60364134/89b37a52-9b78-42a6-9386-0b2d5b5caf33 alt="rounD.gif">
</div>

***

### *[inD](https://arxiv.org/abs/1911.07602)*: The Intersections Drone Dataset

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Automated vehicles rely heavily on data-driven
        methods, especially for complex urban environments. Large
        datasets of real world measurement data in the form of road
        user trajectories are crucial for several tasks like road user
        prediction models or scenario-based safety validation. So far,
        though, this demand is unmet as no public dataset of urban
        road user trajectories is available in an appropriate size, quality
        and variety. By contrast, the highway drone dataset (highD) has
        recently shown that drones are an efficient method for acquiring
        naturalistic road user trajectories. Compared to driving studies
        or ground-level infrastructure sensors, one major advantage of
        using a drone is the possibility to record naturalistic behavior,
        as road users do not notice measurements taking place. Due to
        the ideal viewing angle, an entire intersection scenario can be
        measured with significantly less occlusion than with sensors at
        ground level. Both the class and the trajectory of each road
        user can be extracted from the video recordings with high
        precision using state-of-the-art deep neural networks. Therefore,
        we propose the creation of a comprehensive, large-scale urban
        intersection dataset with naturalistic road user behavior using
        camera-equipped drones as successor of the highD dataset. The
        resulting dataset contains more than 11500 road users including
        vehicles, bicyclists and pedestrians at intersections in Germany
        and is called inD. The dataset consists of 10 hours of measurement
        data from four intersections.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @inproceedings{inDdataset,
        title={The inD Dataset: A Drone Dataset of Naturalistic Road User Trajectories at German Intersections},
        author={Bock, Julian and Krajewski, Robert and Moers, Tobias and Runde, Steffen and Vater, Lennart and Eckstein, Lutz},
        booktitle={2020 IEEE Intelligent Vehicles Symposium (IV)},
        pages={1929-1934},
        year={2020},
        doi={10.1109/IV47402.2020.9304839}
    }
</details>

> #### Dataset Overview
> - Naturalistic trajectory dataset on four different recording locations
> - In total ~ 8,200 vehicles and ~ 5,300 vulnerable road users (VRUs)
> - Road user classes: car, truck/bus, bicycle, pedestrian

<div align="center">
  <img src=https://github.com/westny/dronalize/assets/60364134/98c48e3a-8ac8-4896-863c-c26e08d6764b alt="inD.gif">
</div>

***

### *[exiD](https://ieeexplore.ieee.org/document/9827305)*: The Entries and Exits Drone Dataset

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Development and safety validation of highly automated vehicles increasingly relies on data and data-driven methods. In processing sensor datasets for environment perception, it is common to use public and commercial datasets for training and evaluating machine learning based systems. For system-level evaluation and safety validation of an automated driving system, real-world trajectory datasets are of great value for several tasks in the process, i.a. for testing in simulation, scenario extraction or training of road user agent models. Ground-based recording methods such as sensor-equipped vehicles or infrastructure sensors are sometimes limited, for instance, due to their field of view. Camera-equipped drones, however, offer the ability to record road users without vehicle-to-vehicle occlusion and without influencing traffic. The highway drone dataset (highD) has shown that the recording method is efficient in terms of cumulative kilometers and has become a benchmark dataset for many research questions. It contains many vehicle interactions due to dense traffic, but lacks merging scenarios, which are challenging for highly automated vehicles. Therefore, we propose this highway drone dataset called exiD, recorded using camera-equipped drones at entries and exits on the German Autobahn. The dataset contains 69 172 road users classified as car, truck and vans and a total amount of more than 16 hours of measurement data.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @inproceedings{exiDdataset,
        title={The exiD Dataset: A Real-World Trajectory Dataset of Highly Interactive Highway Scenarios in Germany},
        author={Moers, Tobias and Vater, Lennart and Krajewski, Robert and Bock, Julian and Zlocki, Adrian and Eckstein, Lutz},
        booktitle={2022 IEEE Intelligent Vehicles Symposium (IV)},
        pages={958-964},
        year={2022},
        doi={10.1109/IV51971.2022.9827305}
    }
</details>

> #### Dataset Overview
> - Naturalistic trajectory dataset on seven different recording locations
> - In total ~ 69,430 road users
> - Road user classes: car, truck, bus, motorcycle

<div align="center">
  <img src=https://github.com/user-attachments/assets/69a75f22-2cdf-414d-8238-aefe246f7803 alt="exiD.gif">
</div>

***

### *[uniD](https://levelxdata.com/unid-dataset/)*: The University Drone Dataset

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        The uniD dataset is an innovative collection of naturalistic road user trajectories, captured within the RWTH Aachen University campus using drone technology to address common challenges such as occlusions found in traditional traffic data collection methods. It meticulously documents the movement and classifies each road user by type. Employing cutting-edge computer vision algorithms, the dataset ensures high positional accuracy. Its utility spans various applications, from predicting road user behavior and modeling driver actions to conducting scenario-based safety checks for automated driving systems and facilitating the data-driven design of Highly Automated Driving (HAD) system components.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @misc{uniDdataset,
      title = {{The uniD Dataset: A} university drone dataset},
      author = {{leveLXData}},
      year = {2024},
      howpublished = {\url{https://levelxdata.com/unid-dataset/}},
      note = {Accessed: ...}
    }
</details>

> #### Dataset Overview
> - Naturalistic trajectory dataset on one recording location
> - In total ~ 1,380 vehicles and ~ 8,600 vulnerable road users (VRUs)
> - All road users classes: car, truck/bus, bicycle, pedestrian

<div align="center">
  <img src=https://github.com/user-attachments/assets/ac11c28c-4d25-4d06-8f6d-90d0140065df alt="uniD.gif">
</div>

***

### *[SIND](https://arxiv.org/abs/2209.02297)*: A Drone Dataset at Signalized Intersection in China

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Intersection is one of the most challenging scenarios for autonomous driving tasks. Due to the complexity and stochasticity, essential applications (e.g., behavior modeling, motion prediction, safety validation, etc.) at intersections rely heavily on data-driven techniques. Thus, there is an intense demand for trajectory datasets of traffic participants (TPs) in intersections. Currently, most intersections in urban areas are equipped with traffic lights. However, there is not yet a large-scale, high-quality, publicly available trajectory dataset for signalized intersections. Therefore, in this paper, a typical two-phase signalized intersection is selected in Tianjin, China. Besides, a pipeline is designed to construct a Signalized INtersection Dataset (SIND), which contains 7 hours of recording including over 13,000 TPs with 7 types. Then, the behaviors of traffic light violations in SIND are recorded. Furthermore, the SIND is also compared with other similar works. The features of the SIND can be summarized as follows: 1) SIND provides more comprehensive information, including traffic light states, motion parameters, High Definition (HD) map, etc. 2) The category of TPs is diverse and characteristic, where the proportion of vulnerable road users (VRUs) is up to 62.6% 3) Multiple traffic light violations of non-motor vehicles are shown. We believe that SIND would be an effective supplement to existing datasets and can promote related research on autonomous driving.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @inproceedings{xu2022drone,
      title={{SIND: A} drone dataset at signalized intersection in China},
      author={Xu, Yanchao and Shao, Wenbo and Li, Jun and Yang, Kai and Wang, Weida and Huang, Hua and Lv, Chen and Wang, Hong},
      booktitle={25th International Conference on Intelligent Transportation Systems (ITSC)},
      pages={2471--2478},
      year={2022},
      organization={IEEE}
    }
</details>

> #### Dataset Overview
> - Naturalistic trajectory dataset on four different recording locations
> - Over 13,000 traffic participants of various types
> - Traffic light states included
> - Road user classes: car, truck, bus, motorcycle, tricycle, bicycle, pedestrian

<table>
  <tr>
    <td><img src=https://github.com/user-attachments/assets/d458a2bb-2443-421c-8b54-49a34535651a alt="Chongqing_NR"/></td>
    <td><img src=https://github.com/user-attachments/assets/477693df-3e85-42e0-a317-e69ac16dcc09 alt="Changchun_Pudong"/></td>
    <td><img src=https://github.com/user-attachments/assets/6facdf9f-dbfa-412c-861a-3eda9bb4e4ed alt="Xi'an_Shanglin"/></td>
  </tr>
</table>

<br>

#### Using the SIND Dataset
We provide two options for using the SIND dataset: 1) using the entire dataset, or 2) using the demo dataset available on the [SIND GitHub repository](www.github.com/SOTIF-AVLab/SinD).
Using the demo dataset is recommended for users who want to quickly test the toolbox.
Its use is straightforward, simply name the repository root folder `SIND_demo`, place it in the `datasets` directory, and run the preprocessing script with the `demo` configuration.
> Note that some files in the repo are quite large and may require git-lfs to download properly.


For users who want to use the entire dataset, we ask you to organize the data to match the structure of the other datasets.
Rename the parent folder to `SIND` and make sure it has the following structure:
```
SIND
├── data
│   ├── 6_22_NR_1
│   ├── 6_22_NR_2
│   ├── ...
│   ├── xian_415_n2
│   └── xian_415_n5
└── maps
    ├── Changchun_Pudong.osm
    ├── map_relink_law_save.osm
    ├── NR_ll2.osm
    └── Xi'an_Shanglin.osm
```

> Note that `map_relink_law_save.osm` needs to be downloaded from the GitHub repository.

***

### *[A43](https://data.isac.rwth-aachen.de/?p=58)*: Vehicle Trajectory Dataset from Drone Videos Including Off-Ramp and Congested Traffic

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
        Vehicle trajectory data have become essential for many research fields, such as traffic flow, traffic safety and automated driving. In order to make trajectory data usable for researchers, an overview of the included road section and traffic situation as well as a description of the data processing methodology is necessary. In this paper, we present a trajectory dataset from a German highway with two lanes per direction, an off-ramp and congested traffic in one direction, and an on-ramp in the other direction. The dataset contains 8,648 trajectories and covers 87 minutes and a ~1,200 m long section of the road. The trajectories were extracted from drone videos using a post-trained yolov5 object detection model and projected onto the road surface using a 3D camera calibration. The post-processing methodology can compensate for most false detections and yield accurate speeds and accelerations. We present some applications of the data including a traffic flow analysis and accident risk analysis. The trajectory data are also compared with induction loop data and vehicle-based smartphone sensor data in order to evaluate the plausibility and quality of the trajectory data. The deviations of the speeds and accelerations are estimated at 0.45 m/s and 0.3 m/s2 respectively.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @article{berghaus2024vehicle,
      title={Vehicle trajectory dataset from drone videos including off-ramp and congested traffic--Analysis of data quality, traffic flow, and accident risk},
      author={Berghaus, Moritz and Lamberty, Serge and Ehlers, J{\"o}rg and Kall{\'o}, Eszter and Oeser, Markus},
      journal={Communications in Transportation Research},
      volume={4},
      pages={100133},
      year={2024},
      publisher={Elsevier}
    }
</details>

> #### Dataset Overview
> - Naturalistic trajectory dataset from highway A43 near Münster, Germany
> - In total ~ 8,600 vehicles
> - All road users classes: car, truck, bus, motorcycle 

<table>
  <tr>
    <td><img src=https://github.com/user-attachments/assets/a91f70de-10d5-4ae1-9e13-87d5d2ecf9a5 alt="East"/></td>
  </tr>
    <td><img src=https://github.com/user-attachments/assets/5596b220-7943-44e0-971c-3cb6b2f6d987 alt="West"/></td>
</table>

<br>

# Related work
We have been working with the datasets in several research projects, resulting in multiple published papers focused on behavior prediction.
If you're interested in learning more about our findings, please refer to the following publications:

#### [Diffusion-Based Environment-Aware Trajectory Prediction](https://arxiv.org/abs/2403.11643)
- **Authors:** Theodor Westny, Björn Olofsson, and Erik Frisk
- **Published In:** ArXiv preprint arXiv:2403.11643

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    The ability to predict the future trajectories of traffic participants is crucial for the safe and efficient operation of autonomous vehicles.
    In this paper, a diffusion-based generative model for multi-agent trajectory prediction is proposed.
    The model is capable of capturing the complex interactions between traffic participants and the environment, accurately learning the multimodal nature of the data.
    The effectiveness of the approach is assessed on large-scale datasets of real-world traffic scenarios, showing that our model outperforms several well-established methods in terms of prediction accuracy.
    By the incorporation of differential motion constraints on the model output, we illustrate that our model is capable of generating a diverse set of realistic future trajectories.
    Through the use of an interaction-aware guidance signal, we further demonstrate that the model can be adapted to predict the behavior of less cooperative agents, emphasizing its practical applicability under uncertain traffic conditions.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @article{westny2024diffusion,
      title={Diffusion-Based Environment-Aware Trajectory Prediction},
      author={Westny, Theodor and Olofsson, Bj{\"o}rn and Frisk, Erik},
      journal={arXiv preprint arXiv:2403.11643},
      year={2024}
    }
</details>

#### [MTP-GO: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural ODEs](https://arxiv.org/abs/2302.00735)
- **Authors:** Theodor Westny, Joel Oskarsson, Björn Olofsson, and Erik Frisk
- **Published In:** 2023 IEEE Transactions on Intelligent Vehicles, Vol. 8, No. 9

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    Enabling resilient autonomous motion planning requires robust predictions of surrounding road users' future behavior.
    In response to this need and the associated challenges, we introduce our model titled MTP-GO.
    The model encodes the scene using temporal graph neural networks to produce the inputs to an underlying motion model.
    The motion model is implemented using neural ordinary differential equations where the state-transition functions are learned with the rest of the model.
    Multimodal probabilistic predictions are obtained by combining the concept of mixture density networks and Kalman filtering.
    The results illustrate the predictive capabilities of the proposed model across various data sets, outperforming several state-of-the-art methods on a number of metrics.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @article{westny2023mtp,
      title="{MTP-GO}: Graph-Based Probabilistic Multi-Agent Trajectory Prediction with Neural {ODEs}",
      author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
      journal={IEEE Transactions on Intelligent Vehicles},
      year={2023},
      volume={8},
      number={9},
      pages={4223-4236},
      doi={10.1109/TIV.2023.3282308}}
    }
</details>


#### [Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction](https://arxiv.org/abs/2304.05116)
- **Authors:** Theodor Westny, Joel Oskarsson, Björn Olofsson, and Erik Frisk
- **Published In:** In 2023 IEEE Intelligent Vehicles Symposium (IV)

<details>
  <summary>Abstract</summary>
    <p style="font-style: italic;">
    Given their flexibility and encouraging performance, deep-learning models are becoming standard for motion prediction in autonomous driving.
    However, with great flexibility comes a lack of interpretability and possible violations of physical constraints.
    Accompanying these data-driven methods with differentially-constrained motion models to provide physically feasible trajectories is a promising future direction.
    The foundation for this work is a previously introduced graph-neural-network-based model, MTP-GO.
    The neural network learns to compute the inputs to an underlying motion model to provide physically feasible trajectories.
    This research investigates the performance of various motion models in combination with numerical solvers for the prediction task.
    The study shows that simpler models, such as low-order integrator models, are preferred over more complex, e.g., kinematic models, to achieve accurate predictions.
    Further, the numerical solver can have a substantial impact on performance, advising against commonly used first-order methods like Euler forward.
    Instead, a second-order method like Heun’s can greatly improve predictions.
    </p>
</details>

<details>
  <summary>Bibtex</summary>

    @inproceedings{westny2023eval,
      title={Evaluation of Differentially Constrained Motion Models for Graph-Based Trajectory Prediction},
      author={Westny, Theodor and Oskarsson, Joel and Olofsson, Bj{\"o}rn and Frisk, Erik},
      booktitle={IEEE Intelligent Vehicles Symposium (IV)},
      pages={},
      year={2023},
      doi={10.1109/IV55152.2023.10186615}
    }
</details>

## Contributing
We welcome contributions to the toolbox, and we encourage you to submit pull requests with new features, bug fixes, or improvements.
Any form of collaboration is appreciated, and we are open to suggestions for new features or changes to the existing codebase.
Please direct your inquiries to the authors of the paper.

## Cite
If you use the toolbox in your research, please consider citing the paper:

```
@article{westny2024dronalize,
  title={Toward Unified Practices in Trajectory Prediction Research on Drone Datasets},
  author={Westny, Theodor and Olofsson, Bj{\"o}rn and Frisk, Erik},
  journal={arXiv preprint arXiv:2405.00604},
  year={2024}
}
```

Feel free [email us](mailto:theodor.westny@liu.se) if you have any questions or notice any issues with the toolbox.
If you have any suggestions for improvements or new features, we would be happy to hear from you.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
