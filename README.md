# SAM2_1_fine_tune
The Segment Anything Model 2 ([SAM 2](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints)) is an advanced foundational model designed to tackle prompt-based visual segmentation in both images and videos. 
The model leverages a simple transformer architecture enhanced with streaming memory for optimized processing. SAM 2, trained on a customized dataset, achieves robust performance through targeted fine-tuning techniques.

![model_diagram](https://github.com/user-attachments/assets/a21be4eb-b505-498a-9637-fee70c170e4e)
[source](https://arxiv.org/pdf/2408.00714)

## Description

The key distinction between fine-tuning a model and training one from scratch lies in the initial state of the weights and biases. When training from scratch, these parameters are randomly initialized based on a specific strategy, meaning the model starts with no prior knowledge of the task and performs poorly initially. Fine-tuning, however, begins with pre-existing weights and biases, allowing the model to adapt more effectively to the custom dataset.

The dataset used for fine-tuning SAM 2 consisted of 8-bit RGB images with 50 cm resolution for binary segmentation tasks.

The checkpoint which defined here in the pipeline is `sam2_hiera_large.pt`. 

## Getting Started

### Dependencies
* GDAL, Pytorch- rasterio ... (see installation)
* Cuda-capable GPU [overview here](https://developer.nvidia.com/cuda-gpus)
* Anaconda [download here](https://www.anaconda.com/download)
* developed on Windows 10

### Installation
#### For Windows
```ruby
conda create -n sam2 python=3.11
```
```ruby
conda activate sam2
```
```ruby
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```ruby
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
```
```ruby
cd checkpoints

download_ckpts.sh
```
```ruby
cd ..
cd ../SAM2_fine_tune/environment
```
```ruby
pip install -r requirements.txt
```
#### For Linux
```ruby
conda create -n sam2 python=3.11
```
```ruby
conda activate sam2
```
```ruby
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```ruby
conda install -c conda-forge gdal==3.6
```
```ruby
conda config --env --add channels conda-forge
```
```ruby
conda config --env --set channel_priority strict
```
```ruby
cd ../SAM2_fine_tune/environment
```
```ruby
pip install -r requirements_2.txt
```
## Executing program
set parameters and run in run_pipeline.py

## Authors

* [Benjamin Stöckigt](https://github.com/benjaminstoeckigt)
* [Shadi Ghantous](https://github.com/Shadiouss)

## Acknowledgments

* [segment-anything-2](https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints)
* [Train/Fine-Tune Segment Anything 2 (SAM 2) in 60 lines of code](https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3)


