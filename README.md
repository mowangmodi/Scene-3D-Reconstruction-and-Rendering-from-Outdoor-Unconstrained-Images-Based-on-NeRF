**⚠️ This repository is no longer maintained**

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
-  Python ≥ 3.8 is required. It is recommended to use [Anaconda](https://www.anaconda.com/) for environment management.

- You can create the Conda environment using either of the following methods:

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
⚠️ Note: tiny-cuda-nn require **manual compilation and installation**. Please follow the [official instructions](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) to install it as a PyTorch extension.

# 🔑 Training
## Data download
Download the scenes you want from [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html).

Download the train/test split from [here](https://nerf-w.github.io/) and put under each scene's folder (the **same level** as the "dense" folder).

(**Optional but highly recommended**) Run `python prepare_phototourism.py --root_dir <ROOT_DIR> --img_downscale {an integer, e.g. 2 means half the image sizes} --masks_dir <path_to_masks>` to prepare and save the training data to disk in advance. This step is especially useful if you plan to run multiple experiments or use multiple GPUs, as it **significantly** reduces the data preparation time before training.

Run (example)
```bash 
python prepare_phototourism.py --root_dir <path_to_dataset> --img_downscale <scale_factor> --masks_dir <path_to_masks>
```
# Mask
You can download the corresponding **masks** for each scene from [here](https://drive.google.com/drive/folders/1mat80K8bli-UAAEH4nmkmAkrSTQlcNTG).
### Training model
Run (example)
```bash 
python train.py   --root_dir /path/to/the/datasets/brandenburg_gate/ --img_downscale 2 \
				  --maskrs_max 8e-2 --maskrs_min 6e-3 \
				  --maskrs_k 1e-3 --maskrd 0 \
				  --scale 128 --encode_a \
				  --encode_random --num_gpus 1 \
				  --lr_scheduler steplr --lr 5e-4 \
				  --optimizer radam --masks_dir /path/to/the/Mask/Brandenburg_Gate_Mask \
				  --weightRecA 1e-3 --use_mask
  ```
  Add `--encode_a` to enable the appearance hallucination module, `--use_mask` to enable the anti-occlusion module, and set `--N_vocab` to an integer greater than the number of images (e.g., for "brandenburg_gate" with 1363 images, any number >1363 works). If not set or too small, it will trigger `RuntimeError: CUDA error: device-side assert triggered` from `torch.nn.Embedding`.See [opt.py](https://github.com/mowangmodi/Scene-3D-Reconstruction-and-Rendering-from-Outdoor-Unconstrained-Images-Based-on-NeRF/blob/main/opt.py) for all configurations.
  
 You can monitor the training process by running tensorboard --logdir save/logs/exp_HaNeRF_Brandenburg_Gate --port=8800 and then visiting ```http://localhost:8800``` in your browser.

# Pretrained models 
We provide a **pre-trained model**, which you can download from [here](https://drive.google.com/drive/folders/1mat80K8bli-UAAEH4nmkmAkrSTQlcNTG).
 
#  🔎 Evaluation
Use [eval.py](https://github.com/mowangmodi/Scene-3D-Reconstruction-and-Rendering-from-Outdoor-Unconstrained-Images-Based-on-NeRF/blob/main/eval.py) to inference on all test data. It will create folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the rendered images.
Run (example)
```bash 
python eval.py --root_dir /path/to/the/datasets/brandenburg_gate/ --save_dir save \
               --dataset_name phototourism --scene_name Brandenburg_Gate \
               --split test_test --img_downscale 2 \
               --N_vocab 1500 --scale 128 \
               --chunk 4096 --img_wh 320 240 \
               --encode_a --ckpt_path /path/to/the/brandenburg_gate.ckpt
```
Then you can use [eval_metric.py](https://github.com/mowangmodi/Scene-3D-Reconstruction-and-Rendering-from-Outdoor-Unconstrained-Images-Based-on-NeRF/blob/main/eval_metric.py) to get the quantitative report of different metrics based on the rendered images from [eval.py](https://github.com/mowangmodi/Scene-3D-Reconstruction-and-Rendering-from-Outdoor-Unconstrained-Images-Based-on-NeRF/blob/main/eval.py). It will create a file `result.txt` in the folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the metrics.

Run (example)
```bash 
python eval_metric.py --root_dir /path/to/the/datasets/brandenburg_gate/  --save_dir save \
                      --dataset_name phototourism --scene_name Brandenburg_Gate \
                      --split test_test --img_downscale 2 \
                      --img_wh 320 240
```
# Appearance Rendering
Use [hallucinate.py](https://github.com/mowangmodi/Scene-3D-Reconstruction-and-Rendering-from-Outdoor-Unconstrained-Images-Based-on-NeRF/blob/main/hallucinate.py) to play with Ha-NeRF by hallucinating appearance from different scenes `{example_image}` in different views! It will create folder `{save_dir}/hallucination/{scene_name}` and render the hallucinations, finally create a gif out of them.
Run (example)
```bash 
python hallucinate.py --save_dir save \
                      --ckpt_path /path/to/the/brandenburg_gate.ckpt \
                      --chunk 4096 \
                      --example_image artworks \
                      --scene_name artworks_2_gate
```
note：If you want the view-dependent appearance of a single image, you need to modify [phototourism_mask_grid_sample.py](https://github.com/mowangmodi/Scene-3D-Reconstruction-and-Rendering-from-Outdoor-Unconstrained-Images-Based-on-NeRF/blob/main/datasets/phototourism_mask_grid_sample.py).

# Acknowledge
Our code is based on the awesome pytorch implementation of Hallucinated Neural Radiance Fields in the Wild ([Ha-NeRF](https://github.com/rover-xingyu/Ha-NeRF?tab=readme-ov-file)) and Instant Neural Graphics Primitives with a Multiresolution Hash Encoding([Instant-ngp](https://github.com/kwea123/ngp_pl)). We appreciate all the contributors.

      

