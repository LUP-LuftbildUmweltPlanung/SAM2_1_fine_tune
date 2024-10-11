from create_tiles_unet import split_raster
from predict import predict_and_save_tiles
from train import train_func
import time
import torch


# PARAMETERS
Create_tiles = False
Train = True
Predict = False


# Paths
image_path = r"/home/embedding/Data_Center/DataHouse/+DeepLearning_Extern/beschirmung/RGB_UNET_Modell/Daten/all_dataset/Augsburg_2022/Augsburg_2022_tDOP_RGB_8bit_stack_50cm.tif"
mask_path = r"/home/embedding/Data_Center/DataHouse/+DeepLearning_Extern/beschirmung/RGB_UNET_Modell/Daten/all_dataset/Augsburg_2022/augsburg_2022_canopy_cover_binary_50cm.tif"
base_dir = r"/home/embedding/Data_Center/DataHouse/+DeepLearning_Extern/beschirmung/RGB_UNET_Modell/Daten/Augesburg_sam2_test"


# Tile creation parameters
#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image
patch_size = 400
patch_overlap = 0
max_empty = 0.2 # Maximum no data area in created image crops
split = [0.8, 0.2] # split the data into train & valid dataset
class_zero = True # Enable for seperating 0 prediction class from nodata


# Training parameters
base_dir_train = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\test_8bit\trai" # Path where two folders: "img_tiles" & "mask_tiles" existmodel_path = r"/home/embedding/Data_Center/DataHouse/+DeepLearning_Extern/beschirmung/RGB_UNET_Modell/Daten/model_path/test" # to save the trained model
model_path = r"C:\Users\QuadroRTX\Downloads\fine_tune_sam2\canopy_model" # to save the trained model
description = "des"
model_confg = "large" # 'large', 'base_plus', 'small', 'tiny'  which are  4 different pre-trained SAM 2 models
mode = "binary" # binary if the dataset is (0,1) classification, else #"multi-label"
LEARNING_RATE = 1e-5
EPOCHS = 3
VALID_SCENES = 'vali' # the name of the folder where the validation dataset, 'vali' or 'test'
accuracy_metric = 'loss' # "iou" or "loss
save_confusion_matrix = True # A boolean to enable or disable saving the confusion matrix table."
# confusion matrix
num_classes = 2  # Update for the correct number of classes
# Define human-readable class labels
class_labels = ["Background", "Beschirmung"]



# Prediction parameters
predict_path = r"H:\+DeepLearning_Extern\beschirmung\RGB_UNET_Modell\Daten\test_8bit\vali\img_tiles" # define the images path
predict_model = r"C:\Users\QuadroRTX\Downloads\fine_tune_sam2\canopy_model\model_des_best.torch" # the path where the model saved and the name of the model "name.torch"
AOI = "Dingolfing_038" # Area of Interest (AOI). This parameter is used to append the output TIFF file to define the city of the prediction data.
year = "2018" # the year of the prediction data. To append the output TIFF file to define the year.
validation_vision = True # Confusion matrix and classification report figures, Keep merge and regression False to work!
model_confg_predict = "large" # 'large', 'base_plus', 'small', 'tiny'  which are  4 different pre-trained SAM 2 models
merge = True


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
            class_labels=class_labels

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
        )

    end_time = time.time()
    print(f"Operation completed in {(end_time - start_time) / 60:.2f} minutes.")

if __name__ == '__main__':
    main()


