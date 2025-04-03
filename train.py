import numpy as np
import torch
import os
import tifffile as tiff
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from predict import predict_valid
from sklearn.metrics import confusion_matrix, classification_report
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import hydra
import mlflow
import time

def train_func(base_dir_train, model_confg, epoch, model_path, LEARNING_RATE, description,
               mode="binary", class_zero=False, VALID_SCENES="vali", accuracy_metric='iou', save_confusion_matrix=True,
               num_classes=2, class_labels=list,threshold=0.38, version = "sam2_1",  loss_type="dice", register_model= True):
    with mlflow.start_run(run_name=f"sam2_train_{description}") as run:
        # define the run id
        run_id = run.info.run_id

        # start the running time
        start_time = time.time()  # Start timer for total runtime tracking
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
            print("checkpooints", sam2_checkpoint)
            config_dir = os.path.join(current_dir, "sam2/configs", "sam2")
            print("config", config_dir)


        # Verify that the checkpoint and config files exist
        if not os.path.exists(sam2_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found at: {sam2_checkpoint}")

        if not os.path.exists(os.path.join(config_dir, cfg_name)):
            raise FileNotFoundError(f"Config file not found at: {os.path.join(config_dir, cfg_name)}")

        # Automatically define paths to image and mask tiles for training
        IMG_path_train = os.path.join(base_dir_train, "img_tiles")
        Mask_path_train = os.path.join(base_dir_train, "mask_tiles")

        # List of training image files
        train_data = [{"image": os.path.join(IMG_path_train, img_name), "mask": os.path.join(Mask_path_train, img_name)}
                      for img_name in os.listdir(IMG_path_train)]

        # Get the number of TIFF files in training data
        num_train_files = len(train_data)

        # add params for logging
        mlflow_params = {
            "base_dir_train": base_dir_train,
            "model_confg": model_confg,
            "epochs": epoch,
            "model_path": model_path,
            "learning_rate": LEARNING_RATE,
            "description": description,
            "mode": mode,
            "class_zero": class_zero,
            "VALID_SCENES": VALID_SCENES,
            "accuracy_metric": accuracy_metric,
            "save_confusion_matrix": save_confusion_matrix,
            "num_classes": num_classes,
            "threshold": threshold,
            "version": version,
            "loss_type": loss_type,
            "checkpoint": checkpoint,
            "cfg_name": cfg_name,
            "IMG_path_train": IMG_path_train,
            "Mask_path_train": Mask_path_train,
            "num_train_files": num_train_files
        }

        mlflow_params["class_labels"] = ",".join(class_labels)
        # log params
        mlflow.log_params(mlflow_params)
        # log tages
        mlflow.set_tags({
            "config": model_confg,
            "experiment": description,
            "loss": loss_type,
            "mode": mode
        })


        def read_batch(data, index):
            ent = data[index]
            Img = tiff.imread(ent["image"])

            if Img.shape[-1] == 4:
                Img = Img[:, :, :3]

            if Img.dtype == np.float32 or Img.dtype == np.int32:
                Img = ((Img - Img.min()) / (Img.max() - Img.min()) * 255).astype(np.uint8)

            ann_map = tiff.imread(ent["mask"])
            if class_zero:
                ann_map[ann_map == 1] = 0
                ann_map[ann_map == 2] = 1

            inds = np.unique(ann_map)[1:]
            points = []
            masks = []
            for ind in inds:
                mask = (ann_map == ind).astype(np.uint8)
                masks.append(mask)
                coords = np.argwhere(mask > 0)
                yx = np.array(coords[np.random.randint(len(coords))])
                points.append([[yx[1], yx[0]]])

            return Img, np.array(masks), np.array(points), np.ones([len(masks), 1])

        # Validation step
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=config_dir, version_base='1.2')
        # Load model using the automatically defined paths
        sam2_model = build_sam2(cfg_name, sam2_checkpoint, device="cuda")
        predictor = SAM2ImagePredictor(sam2_model)

        predictor.model.sam_mask_decoder.train(True)
        predictor.model.sam_prompt_encoder.train(True)

        '''
        #The main part of the net is the image encoder, if you have good GPU you can enable training of this part by using:
        predictor.model.image_encoder.train(True)
        #Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
        '''
        optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=LEARNING_RATE, weight_decay=4e-5)
        scaler = torch.cuda.amp.GradScaler() # a more memory-efficient training strategy ( # set mixed precision )

        best_iou = 0
        best_model_path = None
        confusion_matrices = []
        train_ious = []
        training_losses = []
        validation_ious = []
        validation_losses = []

        for itr in range(epoch):
            num_batches = 0
            epoch_mean_iou = 0.0
            epoch_mean_loss = 0.0

            for idx in tqdm(range(num_train_files), desc=f"Epoch {itr + 1}/{epoch}"):
                with torch.cuda.amp.autocast():  # cast to mix precision
                    image, masks, input_points, input_labels = read_batch(train_data, idx) # load data batch
                    if masks.shape[0] == 0: # ignore empty batches
                        continue

                    predictor.set_image(image) # apply SAM image encoder to the image

                  # process the input points using the net prompt encoder
                    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                        input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
                    )
                    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels), boxes=None, masks=None,
                    )

                  # Now that we encoded both the prompt (points) and the image we can finally predict the segmentation masks
                    batched_mode = unnorm_coords.shape[0] > 1 # multi mask prediction
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                         predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=batched_mode,
                        high_res_features=high_res_features,
                    )
                    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1]) # Upscale the masks to the original image resolution

                    if mode == "binary":
                      # Loss functions:  we use the standard cross entropy loss
                        gt_mask = torch.tensor(masks.astype(np.float32)).cuda() # convert the ground truth mask into a torch tensor
                        prd_mask = torch.sigmoid(prd_masks[:, 0]) # Turn logit map to probability map
                        if loss_type== "BCE":
                            # first method  Binary Cross Entropy (BCE) for a sigmoid layer 0, 1
                            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss
                        elif loss_type== "dice":
                            # second one Dice Loss  similar to F1 score (Very good for imbalanced classes)
                            smooth = 1e-5
                            intersection = (gt_mask * prd_mask).sum()
                            dice_loss = 1 - (2. * intersection + smooth) / (gt_mask.sum() + prd_mask.sum() + smooth)
                            seg_loss = dice_loss
                        else:
                            raise ValueError(f"Unsupported loss_type '{loss_type}'. Use 'dice' or 'BCE'.")
                        # comparing the GT mask and the corresponding predicted mask using intersection over union (IOU) metrics
                        # IOU is simply the overlap between the two masks, divided by the combined area of the two masks.
                        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1) # calculate the intersection between the predicted and GT mask, threshold (prd_mask > 0.5) to turn the prediction mask from probability to binary mask.
                        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter) # dividing the intersection by the combined area (union) of the predicted and gt masks
                        # using the IOU as the true score for each mask, and get the score loss as the absolute difference between the predicted scores and the IOU we just calculated
                        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                        loss = seg_loss + score_loss * 0.05 # merge the segmentation loss and score loss

                    else:  # multi-label
                        batch_seg_loss = 0
                        batch_iou_loss = 0
                        for prd_mask, gt_mask, prd_score in zip(prd_masks[:masks.shape[0]], masks, prd_scores[:, 0]):
                            gt_mask = torch.tensor(gt_mask.astype(np.float32)).cuda()
                            prd_mask = torch.sigmoid(prd_mask)
                            if prd_mask.shape != gt_mask.shape:
                                prd_mask = torch.nn.functional.interpolate(prd_mask.unsqueeze(0), size=gt_mask.shape[-2:],
                                                                           mode="bilinear", align_corners=False).squeeze(0)
                            smooth = 1e-5
                            intersection = (gt_mask * prd_mask).sum()
                            dice_loss = 1 - (2. * intersection + smooth) / (gt_mask.sum() + prd_mask.sum() + smooth)
                            batch_seg_loss += dice_loss
                            inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(-2, -1))
                            union = gt_mask.sum(dim=(-2, -1)) + (prd_mask > 0.5).sum(dim=(-2, -1)) - inter
                            iou = inter / (union + 1e-5)
                            score_loss = torch.abs(prd_score - iou).mean()
                            batch_iou_loss += score_loss

                        loss = batch_seg_loss + batch_iou_loss * 0.05
                    # Final step: Backpropogation and saving model
                    predictor.model.zero_grad() # empty gradient
                    scaler.scale(loss).backward() # Backpropagation uses total loss
                    scaler.step(optimizer)
                    scaler.update() # Mix precision

                    num_batches += 1
                    epoch_mean_iou += np.mean(iou.cpu().detach().numpy())
                    # epoch_mean_loss += loss.item() # total loss + iou
                    epoch_mean_loss += seg_loss.item() # just loss values

            epoch_mean_iou /= num_train_files
            epoch_mean_loss /= num_train_files
            train_ious.append(epoch_mean_iou)
            training_losses.append(epoch_mean_loss)
            print(f"Epoch {itr + 1} - Mean IoU: {epoch_mean_iou}, Mean Loss: {epoch_mean_loss}")

            # Validation step
            base_dir_valid = os.path.join(os.path.dirname(base_dir_train), VALID_SCENES)

            # Create the temp_file directory inside base_dir_valid
            temp_file_dir = os.path.join(current_dir, "temp_file")
            if not os.path.exists(temp_file_dir):
                os.makedirs(temp_file_dir)  # Create the temp_file directory if it doesn't exist

            # Create a temporary file in the temp_file directory
            with tempfile.NamedTemporaryFile(dir=temp_file_dir, suffix=".torch", delete=False) as temp_model_file:
                temp_model_path = temp_model_file.name  # Save the path of the temp file

                # Save the model's state dictionary to the temp file
                torch.save(predictor.model.state_dict(), temp_model_path)

                print(f"Temporary file saved at: {temp_model_file.name}")

                mean_valid_iou, mean_valid_loss, mean_total_loss, true_mask_flat, pred_mask_flat = predict_valid(
                    base_dir_valid, temp_model_file.name, mode, model_confg=model_confg, class_zero=class_zero,
                threshold=threshold, version = version)
                validation_ious.append(mean_valid_iou)
                # validation_losses.append(mean_total_loss)
                validation_losses.append(mean_valid_loss)
                print(f"Epoch {itr + 1} - Validation: Mean IoU: {mean_valid_iou}, Mean Loss: {mean_valid_loss}")

                # log metric
                mlflow.log_metric("train_seg_loss", epoch_mean_loss, step=itr)
                mlflow.log_metric("train_iou", epoch_mean_iou, step=itr)
                mlflow.log_metric("val_seg_loss", mean_valid_loss, step=itr)
                mlflow.log_metric("val_iou", mean_valid_iou, step=itr)

            def calculate_accuracy(metric):
                if metric == 'iou':
                    return validation_ious
                elif metric == 'loss':
                    return validation_losses
                else:
                    raise ValueError(f"Unknown accuracy metric: {metric}")

            accuracy = calculate_accuracy(accuracy_metric)
            print(f"Accuracy based on {accuracy_metric}: {np.mean(accuracy):.4f}")

            # Check if this is the best model to save it
            if validation_ious[-1] > best_iou:
                best_iou = validation_ious[-1]
                best_model_path = temp_model_file.name  # Update the best model path
                print("best model path", best_model_path)

        # Save the best model
        if best_model_path:
            final_model_path = os.path.join(model_path, f"model_{description}_best.torch")
            torch.save(torch.load(best_model_path), final_model_path)
            print(f"Best model saved with IOU: {best_iou:.4f}")
            mlflow.log_artifact(final_model_path, artifact_path="models")
            mlflow.set_tag("best_model_iou", best_iou)

            # log the model as mlflow model
            mlflow.pytorch.log_model(
                predictor.model,
                artifact_path="models",
                registered_model_name=description
            )

            # register it
            if register_model:
                mlflow.register_model(
                    model_uri=f"runs:/{run.info.run_id}/models",
                    name=description
                )
            else:
                print("The MLflow's didn't registry the model with version number!")

        # Delete all files in the temp_file directory
        temp_dir = "temp_file"
        for temp_file in os.listdir(temp_dir):
            temp_file_path = os.path.join(temp_dir, temp_file)
            try:
                if os.path.isfile(temp_file_path):
                    os.remove(temp_file_path)
            except Exception as e:
                print(f"Error deleting file {temp_file_path}: {e}")

        # Save the loss and IoU metrics to CSV files at the end of training
        if accuracy_metric == 'iou':
            acc = np.array(validation_ious, dtype=float)  # Convert to float if needed
            accuracy_column_name = 'validation_ious'  # Set the appropriate column name
        elif accuracy_metric == 'loss':
            acc = np.array(validation_losses, dtype=float)  # Convert to float if needed
            accuracy_column_name = 'validation_losses'  # Set the appropriate column name
        else:
            raise ValueError(f"Unknown accuracy metric: {accuracy_metric}")

        metrics_data = {
            'train_loss': training_losses,
            'train_iou': train_ious,
            accuracy_column_name: acc  # The selected metric (IoU, loss, or others)
        }

        metrics_df = pd.DataFrame(metrics_data)

        metrics_csv_path = os.path.join(model_path, f"metrics_{description}.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Metrics saved to {metrics_csv_path}")

        plot_path = os.path.join(model_path, f"metrics_{description}_{accuracy_metric}_comparison.png")

        # Plot comparison between validation and training based on the selected metric
        if accuracy_metric == 'iou':
            # Plot IoU comparison
            plt.figure(figsize=(10, 6))
            plt.plot(train_ious, label='Train IoU')
            plt.plot(validation_ious, label='Validation IoU')
            plt.title('Train vs Validation IoU')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"IoU comparison plot saved at {plot_path}")
            mlflow.log_artifact(plot_path, artifact_path="figures")
            plt.show()


        elif accuracy_metric == 'loss':
            # Plot Loss comparison
            plt.figure(figsize=(10, 6))
            plt.plot(training_losses, label='Train Loss')
            plt.plot(validation_losses, label='Validation Loss')
            plt.title('Train vs Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"Loss comparison plot saved at {plot_path}")
            mlflow.log_artifact(plot_path, artifact_path="figures")
            plt.show()

        # confusion matrix
        num_classes = num_classes  # Update for the correct number of classes

        # Define human-readable class labels
        class_labels = class_labels

        if save_confusion_matrix:
            # Parent directory for saving the results
            parent_dir = os.path.dirname(base_dir_valid)

            # File paths
            confusion_matrix_path_csv = os.path.join(parent_dir, 'confusion_matrix.csv')
            classification_report_path = os.path.join(parent_dir, 'classification_report.csv')
            confusion_matrix_plot_path = os.path.join(parent_dir, 'confusion_matrix_plot.png')
            classification_report_plot_path = os.path.join(parent_dir, 'classification_report_plot.png')

            # Compute the confusion matrix
            cm = confusion_matrix(true_mask_flat, pred_mask_flat, labels=list(range(num_classes)))

            # Convert the confusion matrix to a DataFrame for saving as CSV
            cm_df = pd.DataFrame(cm, index=[f"Actual {label}" for label in class_labels],
                                 columns=[f"Predicted {label}" for label in class_labels])

            # Save the confusion matrix DataFrame to CSV
            cm_df.to_csv(confusion_matrix_path_csv, index=True)
            mlflow.log_artifact(confusion_matrix_path_csv, artifact_path="figures")
            print(f"Confusion matrix saved to {confusion_matrix_path_csv}")

            # Plot the confusion matrix as a heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            # Save the confusion matrix plot
            plt.savefig(confusion_matrix_plot_path)
            mlflow.log_artifact(confusion_matrix_plot_path, artifact_path="figures")
            print(f"Confusion matrix plot saved to {confusion_matrix_plot_path}")
            plt.close()

            # Generate the classification report with human-readable class names
            class_report = classification_report(true_mask_flat, pred_mask_flat, labels=list(range(num_classes)),
                                                 target_names=class_labels, output_dict=True)

            # Convert the classification report to a DataFrame
            class_report_df = pd.DataFrame(class_report).transpose()

            # Save the classification report DataFrame to CSV
            class_report_df.to_csv(classification_report_path, index=True)
            mlflow.log_artifact(classification_report_path, artifact_path="figures")
            print(f"Classification report saved to {classification_report_path}")

            # Plot the classification report as a heatmap (exclude 'support' row)
            plt.figure(figsize=(12, 8))
            sns.heatmap(class_report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')

            # Set labels and titles
            plt.title('Classification Report')
            plt.yticks(rotation=0)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the classification report plot
            plt.savefig(classification_report_plot_path)
            mlflow.log_artifact(classification_report_plot_path, artifact_path="figures")
            print(f"Classification report plot saved to {classification_report_plot_path}")
            plt.close()

            # Log total runtime
            mlflow.log_metric("total_runtime_min", (time.time() - start_time) / 60)

            # Log this script (optional but useful for reproducibility)
            mlflow.log_artifact("main.py", artifact_path="scripts")
