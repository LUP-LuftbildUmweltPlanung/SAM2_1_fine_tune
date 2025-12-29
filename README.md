# SAM2_1_fine_tune
The Segment Anything Model 2 ([SAM 2](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints)) is an advanced foundational model designed to tackle prompt-based visual segmentation in both images and videos. 
The model leverages a simple transformer architecture enhanced with streaming memory for optimized processing. SAM 2, trained on a customized dataset, achieves robust performance through targeted fine-tuning techniques.

![model_diagram](https://github.com/LUP-LuftbildUmweltPlanung/SAM2_1_fine_tune/blob/main/environment/model_diagram.png)

[source](https://arxiv.org/pdf/2408.00714)

## Description

The key distinction between fine-tuning a model and training one from scratch lies in the initial state of the weights and biases. When training from scratch, these parameters are randomly initialized based on a specific strategy, meaning the model starts with no prior knowledge of the task and performs poorly initially. Fine-tuning, however, begins with pre-existing weights and biases, allowing the model to adapt more effectively to the custom dataset.

The dataset used for fine-tuning SAM 2 consisted of 8-bit RGB images with 50 cm resolution for binary segmentation tasks.


## Getting Started

### Dependencies
* GDAL, Pytorch- rasterio ... (see installation)
* Cuda-capable GPU [overview here](https://developer.nvidia.com/cuda-gpus)
* Anaconda [download here](https://www.anaconda.com/download)
* developed on Windows 10

### Installation
#### Installation Instructions (For Windows)
```bash
# 1. Create and activate the Conda environment
conda create -n sam2_1 python=3.11
conda activate sam2_1

# 2. Install PyTorch and its dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Go to the main folder where the script is located
cd ../sam2_1_fine_tune-main

# 4. Clone the SAM2 repository and rename the folder to avoid conflicts
git clone https://github.com/facebookresearch/sam2.git
ren sam2 sam2_conf

# 5. Change into the 'sam2_conf' directory and copy the 'sam2' folder to the 'sam2_1_fine_tune-main' folder
cd sam2_conf

# 6. Install the SAM2 package in editable mode
pip install -e .

# 7. Navigate to the 'checkpoints' folder and download model checkpoints
cd checkpoints && download_ckpts.sh
cd ../..
cd checkpoints_sam2 && download_ckpts.sh

# 8. Go two directories up and install additional dependencies
cd environment
pip install -r requirements.txt
```

#### Installation Instructions (For Linux)
```bash
# 1. Create and activate the Conda environment
conda create -n sam2_1 python=3.11
conda activate sam2_1

# 2. Install PyTorch and its dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Go to the main folder where the script is located
cd ../sam2_1_fine_tune-main

# 4. Clone the SAM2 repository and rename the folder to avoid conflicts
git clone https://github.com/facebookresearch/sam2.git
mv sam2 sam2_conf

# 5. Change into the 'sam2_conf' directory and copy the 'sam2' folder to the 'sam2_1_fine_tune-main' folder
cd sam2_conf
cp -r sam2 ../sam2/

# 6. Install the SAM2 package in editable mode
pip install -e .

# 7. Navigate to the 'checkpoints' folder and download model checkpoints
cd checkpoints && download_ckpts.sh
cd ../..
cd checkpoints_sam2 && download_ckpts.sh

# 8. Go two directories up and install additional dependencies
cd environment
pip install -r requirements_2.txt
```
## Executing program
set parameters and run in run_pipeline.py

**Note** : to run the script with Mlflow please ask me for **mlflow_config** file

## Authors

* [Shadi Ghantous](https://github.com/Shadiouss)
* [Benjamin Stöckigt](https://github.com/benjaminstoeckigt)


## Acknowledgments

* [segment-anything-2](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints)
* [Train/Fine-Tune Segment Anything 2 (SAM 2) in 60 lines of code](https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3)


