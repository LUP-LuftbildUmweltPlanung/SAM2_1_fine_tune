import csv
import numpy as np
import torch
import os
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from osgeo import gdal
from tqdm import tqdm
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score, f1_score
from tifffile import imread
import imageio.v2 as imageio
import hydra


# function to read the binary masks
def read_mask(image_path):
    """Read a mask image from a TIFF file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    return imageio.imread(image_path)  # Read the TIFF file as an array

# function to read the whole images 
def read_image(image_path):
    """Read and resize image using Pillow."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = Image.open(image_path).convert('RGB')  # Convert to RGB
    return np.array(img)

# Merge all the predicted file function
def merge_files(output_folder, AOI, year):
    """Merge all TIF files in the output folder into one."""
    # Find all TIF files in the output folder
    tif_files = glob.glob(os.path.join(output_folder, "*.tif"))
    # Print matched files for debugging
    print("Files to mosaic:", tif_files)
    # Filter out any .ovr files
    tif_files = [f for f in tif_files if not f.endswith('.ovr')]
    # If no files are found, raise an error
    if not tif_files:
        raise RuntimeError("No TIF files found for merging.")

    # Define the nodata value (can be adjusted as needed)
    nodata_value = None
    # Set GDAL warp options for creating the mosaic
    warp_options = gdal.WarpOptions(format="GTIFF", creationOptions=["COMPRESS=LZW", "TILED=YES"],
                                    dstNodata=nodata_value)
    # Define the output file path for the merged TIF (outside the output_folder)
    parent_folder = os.path.dirname(output_folder)
    output_file_name = os.path.basename(output_folder)
    output_file = os.path.join(parent_folder, f"{output_file_name}_{AOI}_{year}_merged.tif")
    # Perform the merge using GDAL Warp
    gdal.Warp(output_file, tif_files, options=warp_options)
    print(f"Merged file created at: {output_file}")

# calculate classufucation report
def calculate_metrics(pred_masks, gt_masks):
    # Flatten masks for metric calculations
    pred_flat = pred_masks.flatten()
    gt_flat = gt_masks.flatten()

    # Determine if data is binary or multiclass
    unique_labels = np.unique(gt_flat)

    if len(unique_labels) <= 2:  # binary case
        average_method = 'binary'
    else:  # multiclass case
        average_method = 'macro'

    # Calculate metrics using the appropriate average method
    precision = precision_score(gt_flat, pred_flat, average=average_method, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, average=average_method, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, average=average_method, zero_division=0)

    return precision, recall, f1
    
def predict_and_save_tiles(input_folder, model_path, mode="binary", model_confg_predict="large", merge=False,
                           class_zero=False, validation_vision=False, AOI=None, year=None, version = "sam2_1"):
    """Predict canopy cover area for all tiles in a folder and save the results."""
    all_precisions = []
    all_recalls = []
    all_f1s = []
                               
    # Set to the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))  
    if version== "sam2_1":
        # Define the checkpoint and config paths based on model configuration
        if 'large' in model_confg_predict:
            checkpoint = "sam2.1_hiera_large.pt"
            cfg_name = 'sam2.1_hiera_l.yaml'
        elif 'base_plus' in model_confg_predict:
            checkpoint = "sam2.1_hiera_base_plus.pt"
            cfg_name = 'sam2.1_hiera_b+.yaml'
        elif 'small' in model_confg_predict:
            checkpoint = "sam2.1_hiera_small.pt"
            cfg_name = 'sam2.1_hiera_s.yaml'
        elif 'tiny' in model_confg_predict:
            checkpoint = "sam2.1_hiera_tiny.pt"
            cfg_name = 'sam2.1_hiera_t.yaml'

        # Set the paths for checkpoints and config files
        sam2_checkpoint = os.path.join(current_dir, "sam2_conf/checkpoints", checkpoint)
        config_dir = os.path.join(current_dir, "sam2/configs", "sam2.1")
    else: # sam2 
        if 'large' in model_confg_predict:
            checkpoint = "sam2_hiera_large.pt"
            cfg_name = 'sam2_hiera_l.yaml'
        elif 'base_plus' in model_confg_predict:
            checkpoint = "sam2_hiera_base_plus.pt"
            cfg_name = 'sam2_hiera_b+.yaml'
        elif 'small' in model_confg_predict:
            checkpoint = "sam2_hiera_small.pt"
            cfg_name = 'sam2_hiera_s.yaml'
        elif 'tiny' in model_confg_predict:
            checkpoint = "sam2_hiera_tiny.pt"
            cfg_name = 'sam2_hiera_t.yaml'

        sam2_checkpoint = os.path.join(current_dir, "checkpoints_sam2", checkpoint)
        config_dir = os.path.join(current_dir, "sam2/configs", "sam2")

    # Verify that the checkpoint and config files exist
    if not os.path.exists(sam2_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {sam2_checkpoint}")

    if not os.path.exists(os.path.join(config_dir, cfg_name)):
        raise FileNotFoundError(f"Config file not found at: {os.path.join(config_dir, cfg_name)}")

    # Re-initialize Hydra configuration for validation
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_dir, version_base='1.2')

    # Build the SAM2 model using the configuration and checkpoint
    sam2_model = build_sam2(cfg_name, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Load model weights from the provided model path
    predictor.model.load_state_dict(torch.load(model_path, map_location="cuda"))

    # Set the model to evaluation mode
    predictor.model.eval()

    # Automatically create an output folder beside the input folder
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    # Get the parent directory of the input_folder
    parent_folder = os.path.dirname(input_folder)
    # Create the output_folder in the parent directory
    output_folder = os.path.join(parent_folder, f"{model_name}_predict_tiles")
    os.makedirs(output_folder, exist_ok=True)

    # Prediction loop
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for image_file in tqdm(os.listdir(input_folder), desc="Processing images"):
            image_path = os.path.join(input_folder, image_file)
            if not image_file.lower().endswith(('.tif', '.tiff')):
                continue

            # Read image
            image = read_image(image_path)
            if image.dtype == np.float32 or image.dtype == np.int32:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

            # Predict masks for the entire image automatically by not passing any points
            with torch.no_grad():
                predictor.set_image(image)
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    multimask_output=False
                )
            # Check if scores are 1-dimensional and handle accordingly
            if scores.ndim == 1:
                np_scores = scores
            else:
                np_scores = scores[:, 0]

            # Convert scores to numpy if necessary
            if isinstance(np_scores, torch.Tensor):
                np_scores = np_scores.cpu().numpy()

            # Check if the maximum score is below a certain threshold, e.g., 0.001
            if np_scores.max() < 0.38:
                # Boost the scores if they are all very low
                masks = np.zeros_like(masks)
            else:
                # Use the original scores if they are above the threshold
                masks = masks
                #print(f"boosted_scores for image {image_file}: {masks}")
            # Sort masks by boosted scores
            sorted_indices = np.argsort(np_scores)[::-1]
            sorted_masks = masks[sorted_indices]

            # Stitch predicted masks into one segmentation mask
            if sorted_masks.ndim == 3:
                seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
                occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
            else:
                raise ValueError("Unexpected mask dimensions: expected 3D array for masks")

            for i in range(sorted_masks.shape[0]):
                mask = sorted_masks[i].astype(bool)
                if mask.sum() == 0:
                    continue
                if (mask & occupancy_mask).sum() / mask.sum() > 0.15:
                    continue
                mask[occupancy_mask] = False

                if mode == "binary":
                    seg_map[mask] = 1
                else:
                    seg_map[mask] = i + 1
                occupancy_mask |= mask

            # Save the segmentation mask as a TIF file in EPSG:25832
            output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + "_predicted.tif")
            with rasterio.Env(GTIFF_SRS_SOURCE='EPSG'):  # Add this line to set the CRS source to EPSG
                with rasterio.open(image_path) as src:
                    transform, width, height = calculate_default_transform(
                        src.crs, 'EPSG:25832', src.width, src.height, *src.bounds)
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': 'EPSG:25832',
                        'transform': transform,
                        'width': width,
                        'height': height,
                        'count': 1,  # Ensure the output has a single band for binary mask
                        'dtype': 'uint8'  # Ensure the data type is uint8 (suitable for binary data)
                    })

                    with rasterio.open(output_path, 'w', **kwargs) as dst:
                        reproject(
                            source=seg_map,
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs='EPSG:25832',
                            resampling=Resampling.nearest
                        )

            # Calculate metrics if ground truth is provided
            if validation_vision:
                # Replace the last folder name "img_tiles" with "mask_tiles"
                gt_folder = os.path.join(os.path.dirname(input_folder), "mask_tiles")
                gt_path = os.path.join(gt_folder, image_file)
                if os.path.exists(gt_path):
                    gt_mask = read_mask(gt_path)
                    if class_zero:
                        # Transform mask values to 0 and 1 for binary classification
                        gt_mask[gt_mask == 1] = 0  # Set class '1' to '0'
                        gt_mask[gt_mask == 2] = 1  # Set class '2' to '1'
                    precision, recall, f1 = calculate_metrics(seg_map, gt_mask)
                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_f1s.append(f1)

    if merge:
        merge_files(output_folder, AOI=AOI, year=year)
    if validation_vision:
        # If you want to print or return the overall metrics:
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1s)
        # print(f"Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}, Average F1 Score: {avg_f1:.4f}")

        output_folder_m = os.path.join(parent_folder, f"{model_name}_confusion_matrix")
        os.makedirs(output_folder_m, exist_ok=True)
        result_path = os.path.join(output_folder_m, "confusion_matrix.csv")

        # Write the results to the CSV file
        with open(result_path, mode="w", newline="") as file:
            writer = csv.writer(file)

            # Write the header
            writer.writerow(["Image Index", "Precision", "Recall", "F1 Score"])

            # Write the individual metrics for each image
            for idx, (precision, recall, f1) in enumerate(zip(all_precisions, all_recalls, all_f1s)):
                writer.writerow([idx, precision, recall, f1])

            # Write the average metrics
            writer.writerow([])  # Blank line for separation
            writer.writerow(["Average", avg_precision, avg_recall, avg_f1])

        print(f"Metrics saved to {result_path}")

def predict_valid(input_folder, model_path, mode="binary", model_confg=None, class_zero=False, version = "sam2_1"):
    """Predict canopy cover area for all tiles in a folder and save the results."""

    # Adjust current_dir to the correct directory level
    current_dir = os.path.abspath(os.path.dirname(__file__))  # Set to the current directory
    if version== "sam2_1":
        # Define the checkpoint and config paths based on model configuration
        if 'large' in model_confg:
            checkpoint = "sam2.1_hiera_large.pt"
            cfg_name = 'sam2.1_hiera_l.yaml'
        elif 'base_plus' in model_confg:
            checkpoint = "sam2.1_hiera_base_plus.pt"
            cfg_name = 'sam2.1_hiera_b+.yaml'
        elif 'small' in model_confg:
            checkpoint = "sam2.1_hiera_small.pt"
            cfg_name = 'sam2.1_hiera_s.yaml'
        elif 'tiny' in model_confg:
            checkpoint = "sam2.1_hiera_tiny.pt"
            cfg_name = 'sam2.1_hiera_t.yaml'
        else:
            checkpoint = "sam2.1_hiera_large.pt"
            cfg_name = 'sam2.1_hiera_l.yaml'

        # Set the paths for checkpoints and config files
        sam2_checkpoint = os.path.join(current_dir, "sam2_conf/checkpoints", checkpoint)
        config_dir = os.path.join(current_dir, "sam2/configs", "sam2.1")
    else:
        if 'large' in model_confg:
            checkpoint = "sam2_hiera_large.pt"
            cfg_name = 'sam2_hiera_l.yaml'
        elif 'base_plus' in model_confg:
            checkpoint = "sam2_hiera_base_plus.pt"
            cfg_name = 'sam2_hiera_b+.yaml'
        elif 'small' in model_confg:
            checkpoint = "sam2_hiera_small.pt"
            cfg_name = 'sam2_hiera_s.yaml'
        elif 'tiny' in model_confg:
            checkpoint = "sam2_hiera_tiny.pt"
            cfg_name = 'sam2_hiera_t.yaml'


        sam2_checkpoint = os.path.join(current_dir, "checkpoints_sam2", checkpoint)
        config_dir = os.path.join(current_dir, "sam2/configs", "sam2")


    # Verify that the checkpoint and config files exist
    if not os.path.exists(sam2_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {sam2_checkpoint}")

    if not os.path.exists(os.path.join(config_dir, cfg_name)):
        raise FileNotFoundError(f"Config file not found at: {os.path.join(config_dir, cfg_name)}")


    # Re-initialize Hydra configuration for validation
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=config_dir, version_base='1.2')

    # Build the SAM2 model using the configuration and checkpoint
    sam2_model = build_sam2(cfg_name, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)


    # Load the pre-trained weights from the model path
    predictor.model.load_state_dict(torch.load(model_path, map_location="cuda"))
    predictor.model.eval()

    # Define input folders
    img = os.path.join(input_folder, "img_tiles")
    truth_label = os.path.join(input_folder, "mask_tiles")

    # Lists to store metrics
    iou_scores = []
    valid_seg_losses = []
    total_seg_loss = 0
    valid_ious = []

    # Prediction loop
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for image_file in tqdm(os.listdir(img), desc="Processing Validation Set"):
            image_path = os.path.join(img, image_file)
            if not image_file.lower().endswith(('.tif', '.tiff')):
                continue

            # Read image
            image = read_image(image_path)

            if image.dtype == np.float32 or image.dtype == np.int32:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

            # Predict masks for the entire image
            with torch.no_grad():
                predictor.set_image(image)
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    multimask_output=(mode == "multi-label")
                )

            if isinstance(scores, torch.Tensor):
                np_scores = scores.cpu().numpy()
            else:
                np_scores = scores

            # Sort masks by scores
            sorted_indices = np.argsort(np_scores)[::-1]
            sorted_masks = masks[sorted_indices]

            # Stitch predicted masks into one segmentation mask
            if sorted_masks.ndim == 3:
                seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
                occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
            else:
                raise ValueError("Unexpected mask dimensions: expected 3D array for masks")

            for i in range(sorted_masks.shape[0]):
                mask = sorted_masks[i].astype(bool)
                if (mask & occupancy_mask).sum() / mask.sum() > 0.15:
                    continue
                mask[occupancy_mask] = False

                if mode == "binary":
                    seg_map[mask] = 1
                else:
                    seg_map[mask] = i + 1
                occupancy_mask |= mask

            # Load the corresponding ground truth mask
            mask_file = os.path.splitext(image_file)[0] + ".tif"
            mask_path = os.path.join(truth_label, mask_file)
            if os.path.exists(mask_path):
                true_mask = imread(mask_path)

                if class_zero:
                    true_mask[true_mask == 1] = 0
                    true_mask[true_mask == 2] = 1

                # Resize ground truth mask to match predicted segmentation map
                true_mask_resized = resize(true_mask, seg_map.shape, order=0, preserve_range=True).astype(np.uint8)

                ###########################################

                # Ensure both masks are in numpy array format
                if not isinstance(true_mask_resized, np.ndarray):
                    true_mask_np = true_mask_resized.cpu().numpy()  # Convert to numpy if it's a tensor
                else:
                    true_mask_np = true_mask_resized

                if not isinstance(seg_map, np.ndarray):
                    pred_mask_np = seg_map.cpu().numpy()  # Convert to numpy if it's a tensor
                else:
                    pred_mask_np = seg_map

                # Flatten the arrays to compare them for the confusion matrix
                true_mask_flat = true_mask_np.flatten()
                pred_mask_flat = pred_mask_np.flatten()

                # Ensure that both masks are of float type without scaling if they are already binary
                if pred_mask_np.dtype != np.float32:
                    pred_mask_np = pred_mask_np.astype(np.float32)

                if true_mask_np.dtype != np.float32:
                    true_mask_np = true_mask_np.astype(np.float32)

                # Convert NumPy arrays to PyTorch tensors before performing PyTorch operations
                true_mask_tensor = torch.tensor(true_mask_np, dtype=torch.float32).cuda()
                pred_mask_tensor = torch.tensor(pred_mask_np, dtype=torch.float32).cuda()

                # Apply threshold to the predicted mask before calculating IoU
                pred_mask_binary = (pred_mask_tensor > 0.5).float()

                # Calculate IoU using PyTorch tensors
                inter = (true_mask_tensor * pred_mask_binary).sum()
                union = true_mask_tensor.sum() + pred_mask_binary.sum() - inter
                #iou = inter / (union + 1e-5) if union > 0 else 0
                iou = inter / (union + 1e-5) if union > 0 else torch.tensor(0.0).cuda()
                iou_scores.append(iou)

                # Calculate segmentation loss (binary cross-entropy) using the thresholded binary mask
                # seg_loss = (-true_mask_tensor * torch.log(pred_mask_binary + 1e-5) -
                #              (1 - true_mask_tensor) * torch.log((1 - pred_mask_binary) + 1e-5)).mean()

                smooth = 1e-5  # Add a small value to avoid division by zero
                intersection = (true_mask_tensor * pred_mask_binary).sum()
                dice_loss = 1 - (2. * intersection + smooth) / (
                            true_mask_tensor.sum() + pred_mask_binary.sum() + smooth)

                # Replace the segmentation loss with dice loss
                seg_loss = dice_loss

                # Calculate score loss
                prd_score_tensor = torch.tensor(np_scores[sorted_indices]).cuda()
                score_loss = torch.abs(prd_score_tensor - iou).mean()

                # Calculate total loss
                loss = seg_loss + score_loss * 0.05

                # Save metrics for comparison
                valid_seg_losses.append(seg_loss.item())
                # print("seg_loss", seg_loss.item())
                valid_ious.append(iou.item())
                # print("iou", iou.item())

                total_seg_loss += seg_loss.item()

            else:
                print(f"Ground truth mask not found for image {image_file}. Skipping IoU calculation.")

    # Compute mean IoU across all images (Move tensors to CPU before converting to NumPy)
    mean_iou = np.mean([iou.cpu().numpy() for iou in iou_scores]) if iou_scores else 0

    # Calculate mean segmentation loss (Move tensors to CPU before converting to NumPy)
    mean_seg_loss = total_seg_loss / len(iou_scores) if iou_scores else 0

    # If you switch back to training later, ensure to set the model back
    predictor.model.train()
    return mean_iou, mean_seg_loss, true_mask_flat, pred_mask_flat
