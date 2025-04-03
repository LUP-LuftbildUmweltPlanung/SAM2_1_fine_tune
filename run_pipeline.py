import os
from create_tiles_unet import split_raster
from predict import predict_and_save_tiles
from train import train_func
import time
import torch
import warnings
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow_config import *

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['GDAL_CACHEMAX'] = '512'  # Set to 1024 MB (adjust based on available RAM)

# set Mlflow request
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "300"

# Initialize Mlflow Client
client = MlflowClient()

# Define Experiment name
experiment_name = "Fine_Tune_Segment_Anything2"

# check if experiment Exists
experiment = client.get_experiment_by_name(experiment_name)

# create new exp if not found
if experiment is None:
    print(f" Experiment '{experiment_name}' not found! Creating a new one...")
    experiment_id = client.create_experiment(name=experiment_name)
    print(f" Created new experiment: {experiment_name} (ID: {experiment_id})")
else:
    experiment_id = experiment.experiment_id
    print(f" Using existing experiment: {experiment_name} (ID: {experiment_id})")

# set the Active Experiment
mlflow.set_experiment(experiment_name)

# confirm Artifact Location:
experiment = client.get_experiment(experiment_id)
print(f" Experiment '{experiment_name}' Artifact Location: {experiment.artifact_location}")





# PARAMETERS
Create_tiles = False
Train = False
Predict = True
version = "sam2_1" # define the version of sam "sam_2" or "sam2_1"

# Paths
image_path = r"PATH"
mask_path = r"PATH"
base_dir = r"PATH" # define the path to save all the created tiles


# Tile creation parameters
#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image
patch_size = 400
patch_overlap = 0
max_empty = 0.2 # Maximum no data area in created image crops
split = [0.8, 0.2] # split the data into train & valid dataset
class_zero = True # Enable for seperating 0 prediction class from nodata


# Training parameters
base_dir_train = r"PATH" # Path where two folders: "img_tiles" & "mask_tiles"
model_path = r"PATH" # to save the trained model
description = "Canopy_mdoel" # give the name of saved model. For example "model_Canopy_mdoel_best.torch"
model_confg = "large" # 'large', 'base_plus', 'small', 'tiny'  which are  4 different pre-trained SAM 2 models
mode = "binary" # binary if the dataset is (0,1) classification, else #"multi-label"
LEARNING_RATE = 1e-5
EPOCHS = 3
VALID_SCENES = 'vali' # the name of the folder where the validation dataset, 'vali' or 'test' if you create three different folders in create tiles part
threshold = 0.38      # Classification threshold
accuracy_metric = 'loss' # "iou" or "loss
loss_type="dice" # BCE
save_confusion_matrix = True # A boolean to enable or disable saving the confusion matrix table."
num_classes = 2  # Update for the correct number of classes
class_labels = ["Background", "Beschirmung"] # Define human-readable class labels
register_model = False



# Prediction parameters
predict_path = r"PATH" # define the images path
predict_model = r"PATH" # the path where the model saved and the name of the model "model_Canopy_mdoel_best.torch"
AOI = "Dingolfing" # Area of Interest (AOI). This parameter is used to append the output TIFF file to define the city of the prediction data.
year = "2018" # the year of the prediction data. To append the output TIFF file to define the year.
validation_vision = True # Confusion matrix and classification report figures, Keep merge and regression False to work!
model_confg_predict = "large" # 'large', 'base_plus', 'small', 'tiny'  which are  4 different pre-trained SAM 2 models
merge = True # merge all the predicted tiles in one ratser file


def main():
    start_time = time.time()

    if torch.cuda.is_available():
        print("CUDA device is available.")
    else:
        print("No CUDA device available, running on CPU.")

    if Create_tiles:
        split_raster(
            path_to_raster=image_path,
            path_to_mask=mask_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            base_dir=base_dir,
            split=split,
            max_empty=max_empty,
            class_zero=class_zero
        )

    if Train:
        train_func(
            base_dir_train=base_dir_train,
            model_confg=model_confg,
            epoch=EPOCHS,
            LEARNING_RATE=LEARNING_RATE,
            model_path=model_path,
            description=description,
            mode=mode,
            class_zero=class_zero,
            VALID_SCENES=VALID_SCENES,
            accuracy_metric = accuracy_metric,
            save_confusion_matrix = save_confusion_matrix,
            num_classes=num_classes,
            class_labels=class_labels,
            threshold=threshold,
            version=version,
            loss_type=loss_type,
            register_model= register_model

        )

    if Predict:
        predict_and_save_tiles(
            input_folder=predict_path,
            model_path=predict_model,
            mode=mode,
            model_confg_predict=model_confg_predict,
            merge=merge,
            class_zero=class_zero,
            validation_vision=validation_vision,
            AOI=AOI,
            year=year,
            threshold=threshold,
            version=version,
            loss_type=loss_type
        )

    end_time = time.time()
    print(f"Operation completed in {(end_time - start_time) / 60:.2f} minutes.")

if __name__ == '__main__':
    main()


