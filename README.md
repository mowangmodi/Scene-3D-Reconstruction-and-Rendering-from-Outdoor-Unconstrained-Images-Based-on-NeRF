

# 💻 Installation
This implementation only includes libraries that have been actually tested and used, along with their corresponding hardware and software environments. Components that were not used are not listed here.
# Hardware
- **OS**: Ubuntu 20.04  
- **GPU**: NVIDIA GPU with Compute Capability ≥ 8.6 and memory > 8GB  *(Tested on RTX 4090 with CUDA 11.7; other versions may also be compatible))* 
- **RAM**: 128GB 
#  Software

Make sure you have Git installed. Then run the following command to clone the repository:

```bash
git clone https://github.com/mowangmodi/Scene-3D-Reconstruction-and-Rendering-from-OutdoorUnconstrained-Images-Based-on-NeRF.git
```
- : Python ≥ 3.8 is required. It is recommended to use [Anaconda](https://www.anaconda.com/) for environment management.

- :You can create the Conda environment using either of the following methods:

```bash
# Method 1: Create manually
conda create -n nerf python=3.8

# Method 2: Create from environment.yaml (if provided)
conda env create -f environment.yaml

# Activate the environment
conda activate nerf
# Cuda extension
Please make sure your `pip` version is upgraded to 22.1 or higher before installing the CUDA extension.
Run the following command to install the CUDA extension (please run this every time after pulling new code):
pip install -U pip
pip install models/csrc/
```
# 🔑 Training
## Data download
Download the scenes you want from [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html)
Download the train/test split from [here](https://nerf-w.github.io/) and put under each scene's folder (the **same level** as the "dense" folder)
(**Optional but highly recommended**) Run `python prepare_phototourism.py --root_dir <ROOT_DIR> --img_downscale {an integer, e.g. 2 means half the image sizes} --masks_dir <path_to_masks>` to prepare and save the training data to disk in advance. This step is especially useful if you plan to run multiple experiments or use multiple GPUs, as it **significantly** reduces the data preparation time before training.

Run (example)
```bash 
python prepare_phototourism.py --root_dir <path_to_dataset> --img_downscale <scale_factor> --masks_dir <path_to_masks>
```
