import os
import shutil
import warnings
from glob import glob
from pathlib import Path
import json
import numpy as np
import rasterio
import slidingwindow
from osgeo import gdal


def delete_folder(folder_path):
    """Deletes an empty folder by given path"""
    # checking whether folder exists or not
    if os.path.exists(folder_path):

        # checking whether the folder is empty or not
        if len(os.listdir(folder_path)) == 0:
            # removing the file using the os.remove() method
            os.rmdir(folder_path)
        else:
            # messaging saying folder not empty
            print("Folder is not empty")
    else:
        # file not found message
        print("Folder not found in the directory")


def compute_windows(numpy_image, patch_size, patch_overlap):
    """
    Create a sliding window object from a raster tile.

    Parameters:
    -----------
        numpy_image :   Raster object as numpy array to cut into crops
        patch_size :    Size of output crops
        patch_overlap : Overlap between crops

    Returns:
    ---------
        windows : a sliding windows object

    References:
    ----------
        https://deepforest.readthedocs.io/en/latest/_modules/deepforest/preprocess.html
    """
    if patch_overlap > 1:
        raise ValueError(f"Patch overlap {patch_overlap} must be between 0 - 1")

    # Generate overlapping sliding windows
    windows = slidingwindow.generate(numpy_image,
                                     slidingwindow.DimOrder.HeightWidthChannel,
                                     patch_size, patch_overlap)

    return windows


def get_files(directory, file_type):
    """Returns a list of all files of the given type in the given directory."""
    directory = Path(directory)
    ori_dir = os.getcwd()
    os.chdir(directory)
    files = [directory / file for file in glob('*.' + file_type)]
    os.chdir(ori_dir)
    return files


def create_train_test_split(path, split=None):
    """
    Creates a train/test/vali split on image files split between two directories (images, masks) in the provided path.
    Resulting split is stored in the same directory.

    Parameters:
    -----------
        path :  Path containing the directiories.
        split : Split ratio (default=None -> [0.7, 0.2, 0.1])
    """
    if split is None:
        split = [0.7, 0.2, 0.1]
    if np.round(np.sum(split), decimals=3) != 1.0:
        split = [0.7, 0.2, 0.1]
        warnings.warn('Train/Vali/Test-Split percentage does not sum to 1, reseting to 70%/20%/10%.')

    source = Path(path)
    sources = [p.path for p in os.scandir(str(source)) if p.is_dir()]

    Path(str(source) + r'\trai\mask_tiles').mkdir(parents=True, exist_ok=True)
    Path(str(source) + r'\trai\img_tiles').mkdir(parents=True, exist_ok=True)
    Path(str(source) + r'\vali\mask_tiles').mkdir(parents=True, exist_ok=True)
    Path(str(source) + r'\vali\img_tiles').mkdir(parents=True, exist_ok=True)
    if split[-1] != 0 and len(split) == 3:
        Path(str(source) + r'\test\mask_tiles').mkdir(parents=True, exist_ok=True)
        Path(str(source) + r'\test\img_tiles').mkdir(parents=True, exist_ok=True)

    s = sources[0]

    files = get_files(s, 'tif')
    np.random.shuffle(files)
    train_files = files[:int(len(files) * split[0])]
    if split[-1] == 0 or len(split) == 2:
        vali_files = files[int(len(files) * split[0]):]
    else:
        vali_files = files[int(len(files) * split[0]):int(len(files) * np.sum(split[:2]))]
        test_files = files[int(len(files) * np.sum(split[:2])):]

    storage = [str(file).rsplit('\\', 1)[-1] for file in files]

    train_storage = [sources[1] + '\\' + mask_file for mask_file in storage[:int(len(files) * split[0])]]
    if split[-1] == 0 or len(split) == 2:
        vali_storage = [sources[1] + '\\' + mask_file for mask_file in storage[int(len(files) * split[0]):]]
    else:
        vali_storage = [sources[1] + '\\' + mask_file for mask_file in
                        storage[int(len(files) * split[0]):int(len(files) * np.sum(split[:2]))]]
        test_storage = [sources[1] + '\\' + mask_file for mask_file in
                        storage[int(len(files) * np.sum(split[:2])):]]

    train_files += train_storage
    vali_files += vali_storage
    if split[-1] != 0 and len(split) == 3:
        test_files += test_storage

    for f in train_files:
        if str(f).rsplit('\\', 1)[0].endswith('img_tiles'):
            dest = Path(str(source) + r'\trai\img_tiles')
        else:
            dest = Path(str(source) + r'\trai\mask_tiles')

        try:
            os.rename(f, dest / (str(f).rsplit('\\', 1)[-1]))

        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")

    for f in vali_files:
        if str(f).rsplit('\\', 1)[0].endswith('img_tiles'):
            dest = Path(str(source) + r'\vali\img_tiles')
        else:
            dest = Path(str(source) + r'\vali\mask_tiles')

        try:
            os.rename(f, dest / (str(f).rsplit('\\', 1)[-1]))

        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")

        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")

    if split[-1] != 0 and len(split) == 3:
        for f in test_files:
            if str(f).rsplit('\\', 1)[0].endswith('img_tiles'):
                dest = Path(str(source) + r'\test\img_tiles')
            else:
                dest = Path(str(source) + r'\test\mask_tiles')

            try:
                os.rename(f, dest / (str(f).rsplit('\\', 1)[-1]))

            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")

            # If there is any permission issue
            except PermissionError:
                print("Permission denied.")

    delete_folder(Path(str(source) + r'\img_tiles'))
    delete_folder(Path(str(source) + r'\mask_tiles'))


def save_crop(base_dir, image_name, index, crop, crop_mask, bands_img, rect, geotrans, geoproj, raster_dtype,
              mask_dtype):
    """
    Save window crop as image file to be read by PIL. Filename should match the image_name + window index.

    Parameters:
    -----------
        base_dir : Directory in which to store image and mask
        image_name : Name of the image file
        index : Index of the image file
        crop : Cropped image file
        crop_mask : Cropped corresponding mask (can be None)
        bands_img : Bands of the image
        rect : Something necessary for the geotransformation
        geotrans : Geotransformation data of the image file
        geoproj : Geoprojection data of the image file
        raster_dtype : image file datatype
        mask_dtype : Mask file datatype
        quantile_stretch : If a 99% quantile stretch should be performed (default=False)
    """
    include_mask = crop_mask is not None

    # create dir if needed
    if include_mask and not os.path.exists(base_dir + "/mask_tiles"):
        os.makedirs(base_dir + "/mask_tiles")
    if not os.path.exists(base_dir + "/img_tiles"):
        os.makedirs(base_dir + "/img_tiles")
    image_basename = os.path.splitext(image_name)[0]

    driver = gdal.GetDriverByName('GTiff')
    if raster_dtype.endswith("int16"):
        out_ds = driver.Create("{}/{}_{}.tif".format(base_dir + "\\img_tiles", image_basename, index), crop.shape[0],
                               crop.shape[1], bands_img, gdal.GDT_UInt16)
        raster_dtype_factor = 65536
    elif raster_dtype.endswith("int8"):
        out_ds = driver.Create("{}/{}_{}.tif".format(base_dir + "\\img_tiles", image_basename, index), crop.shape[0],
                               crop.shape[1], bands_img, gdal.GDT_Byte)
        raster_dtype_factor = 256
    elif raster_dtype.endswith("float32"):
        out_ds = driver.Create("{}/{}_{}.tif".format(base_dir + "\\img_tiles", image_basename, index), crop.shape[0],
                               crop.shape[1], bands_img, gdal.GDT_Float32)

    else:
        print("raster_dtype error:" + str(raster_dtype))

    xmin, ymax, xres, yres = rect
    out_ds.SetGeoTransform(
        [xmin * geotrans[1] + geotrans[0], geotrans[1], 0, geotrans[3] - ymax * geotrans[1], 0, geotrans[5], ])
    out_ds.SetProjection(geoproj)
    for i in range(bands_img):
        out_ds.GetRasterBand(i + 1).WriteArray(crop[:, :, i])

    out_ds.FlushCache()
    del out_ds

    if include_mask:
        driver2 = gdal.GetDriverByName('GTiff')
        if "float" in mask_dtype:
            out_ds2 = driver2.Create("{}/{}_{}.tif".format(base_dir + "\\mask_tiles", image_basename, index),
                                     crop_mask.shape[0], crop_mask.shape[1], 1, gdal.GDT_Float32)
        else:
            out_ds2 = driver2.Create("{}/{}_{}.tif".format(base_dir + "\\mask_tiles", image_basename, index),
                                     crop_mask.shape[0], crop_mask.shape[1], 1, gdal.GDT_Byte)
        out_ds2.SetGeoTransform(
            [xmin * geotrans[1] + geotrans[0], geotrans[1], 0, geotrans[3] - ymax * geotrans[1], 0, geotrans[5], ])
        out_ds2.SetProjection(geoproj)
        out_ds2.GetRasterBand(1).WriteArray(crop_mask[:, :, 0])

        out_ds2.FlushCache()

        del out_ds2


def split_raster(path_to_raster=None,
                 path_to_mask=None,
                 base_dir=".",
                 patch_size=400,
                 patch_overlap=0.20,
                 split=None,
                 max_empty=0.9,
                 class_zero=False):
    """
    Divide a large tile into smaller arrays. Each crop will be saved to file.
    For not perfectly overlapping raster size, the overlapping area will be used (assumes roughly similar pixel size).

    Parameters:
    -----------
        path_to_raster: Path to a image tile that can be read by rasterio on disk
        path_to_mask: Path to a corresponding mask tile that can be read by rasterio on disk
        base_dir : Where to save the annotations and image crops
        patch_size: Maximum dimensions of square window
        patch_overlap: Percent of overlap among windows 0->1
        quantile_stretch: If True, perform a 99% quantile stretch on the image data (default=False)
        split: Split of training/testing/validation data (default=None -> [0.7, 0.2, 0.1])

    References:
    ----------
        https://deepforest.readthedocs.io/en/latest/_modules/deepforest/preprocess.html#split_raster
    """
    if split is None:
        split = [0.7, 0.2, 0.1]
    include_mask = path_to_mask is not None

    numpy_image = rasterio.open(path_to_raster).read()
    bands_img = numpy_image.shape[0]
    raster_dtype = str(rasterio.open(path_to_raster).dtypes[0])

    # setnodata 0
    nodata = rasterio.open(path_to_raster).nodata

    out_l, out_w, out_o1, out_t, out_o2, out_h = gdal.Open(path_to_raster).GetGeoTransform()

    if include_mask:
        img_l, img_w, _, img_t, _, img_h = gdal.Open(path_to_raster).GetGeoTransform()
        msk_l, msk_w, _, msk_t, _, msk_h = gdal.Open(path_to_mask).GetGeoTransform()
        img_w = np.around(img_w, decimals=3)
        img_h = np.around(img_h, decimals=3)
        msk_w = np.around(msk_w, decimals=3)
        msk_h = np.around(msk_h, decimals=3)

        mask_dtype = str(rasterio.open(path_to_mask).dtypes[0])
        numpy_image_mask = rasterio.open(path_to_mask).read()
        nodata_mask = rasterio.open(path_to_mask).nodata
        if class_zero:
            numpy_image_mask[numpy_image_mask != nodata_mask] += 1
            #print(np.unique(numpy_image_mask))

        if np.round(img_l, decimals=3) != np.round(msk_l, decimals=3) \
                or np.round(img_t, decimals=3) != np.round(msk_t, decimals=3) \
                or numpy_image.shape[1:] != numpy_image_mask.shape[1:]:
            print('Image and mask sizes do not match. Performing adjustments... ')
            out_l = np.max([img_l, msk_l])
            out_t = np.min([img_t, msk_t])

            img_range = np.array([[img_l, img_l + img_w * numpy_image.shape[2]],
                                  [img_t + img_h * numpy_image.shape[1], img_t]])

            msk_range = np.array([[msk_l, msk_l + msk_w * numpy_image_mask.shape[2]],
                                  [msk_t + msk_h * numpy_image_mask.shape[1], msk_t]])

            w_offset = np.around((img_l / img_w % 1 - msk_l / msk_w % 1) * msk_w, decimals=3)
            h_offset = np.around((img_t / img_h % 1 - msk_t / msk_h % 1) * msk_h, decimals=3)

            if w_offset > 0.5 * np.absolute(msk_w):
                w_offset -= np.absolute(msk_w)
            elif w_offset <= -0.5 * np.absolute(msk_w):
                w_offset += np.absolute(msk_w)

            if h_offset > 0.5 * np.absolute(msk_h):
                h_offset -= np.absolute(msk_h)
            elif h_offset <= -0.5 * np.absolute(msk_h):
                h_offset += np.absolute(msk_h)

            msk_range[0] += w_offset
            msk_range[1] += h_offset

            out_range = np.array([[np.max(np.array([img_range, msk_range])[:, 0, 0]),
                                   np.min(np.array([img_range, msk_range])[:, 0, 1])],
                                  [np.max(np.array([img_range, msk_range])[:, 1, 0]),
                                   np.min(np.array([img_range, msk_range])[:, 1, 1])]])

            img_adj = out_range - img_range
            img_adj[0] /= img_w
            img_adj[1] = img_adj[1, ::-1] / img_h
            img_adj = np.round(img_adj[[1, 0]])  # if error, replace with img_adj.round()
            img_adj[:, 1] += np.array(numpy_image.shape[1:])
            img_adj = img_adj.astype(int)

            msk_adj = out_range - msk_range
            msk_adj[0] /= msk_w
            msk_adj[1] = msk_adj[1, ::-1] / msk_h
            msk_adj = np.round(msk_adj[[1, 0]])  # if error, replace with msk_adj.round()
            msk_adj[:, 1] += np.array(numpy_image_mask.shape[1:])
            msk_adj = msk_adj.astype(int)

            numpy_image = numpy_image[:, img_adj[0, 0]:img_adj[0, 1], img_adj[1, 0]:img_adj[1, 1]]
            numpy_image_mask = numpy_image_mask[:, msk_adj[0, 0]:msk_adj[0, 1], msk_adj[1, 0]:msk_adj[1, 1]]

            assert numpy_image.shape[1:] == numpy_image_mask.shape[1:], "Some issue with the adjustments"
            print(f'Done! Adjusted images new size is {numpy_image.shape[1:]}.\n')

        no_data_values = np.sum(numpy_image_mask[0, :, :] == nodata_mask)
        no_data_percentage = round((no_data_values / len(numpy_image_mask[0].flatten())) * 100)
        no_data_values_image = np.sum(numpy_image[0, :, :] == nodata)
        no_data_percentage_image = round((no_data_values_image / len(numpy_image[0].flatten())) * 100)

        if no_data_values:
            print(
                f'{no_data_values} no-data-pixels found in mask ({no_data_percentage}%), setting parts of image to 0.')
        if no_data_values_image:
            print(
                f'{no_data_values_image} no-data-pixels found in image ({no_data_percentage_image}%), setting parts of mask to 0.')

        # Create a boolean mask where any of the image or mask bands has a nodata value
        nodata_mask = (numpy_image == nodata).any(axis=0) | (numpy_image_mask == nodata_mask).any(axis=0)

        # Set each band of the image and mask to 0 where the mask is True
        numpy_image[:, nodata_mask] = 0
        numpy_image_mask[:, nodata_mask] = 0

        numpy_image_mask2 = np.moveaxis(numpy_image_mask, 0, 2)

    else:
        # if no mask is included
        # Create a boolean mask where any of the image bands has a nodata value
        nodata_mask = (numpy_image == nodata).any(axis=0)

        # Set each band of the image and mask to 0 where the mask is True
        numpy_image[:, nodata_mask] = 0

    numpy_image2 = np.moveaxis(numpy_image, 0, 2)

    geotrans = (out_l, out_w, out_o1, out_t, out_o2, out_h)
    geoproj = gdal.Open(path_to_raster).GetProjection()

    # Check if patch size is greater than image size
    height = numpy_image2.shape[0]
    width = numpy_image2.shape[1]

    if any(np.array([height, width]) < patch_size):
        raise ValueError("Patch size of {} is larger than the image dimensions {}".format(
            patch_size, [height, width]))

    # Compute sliding window index
    windows = compute_windows(numpy_image2, patch_size, patch_overlap)

    # Get image name for indexing
    image_name = os.path.basename(path_to_raster)

    for index, window in enumerate(windows):
        # Crop image
        crop = numpy_image2[windows[index].indices()]

        if crop.size == 0:
            continue
        if np.sum(crop != 0) < np.prod(crop.shape) * (1 - max_empty):
            continue

        if include_mask:
            crop_mask = numpy_image_mask2[windows[index].indices()]
            # skip if empty crop
            if crop_mask.size == 0:
                continue
            if np.sum(crop_mask != 0) < np.prod(crop_mask.shape) * (1 - max_empty):
                continue
        else:
            crop_mask = None
            mask_dtype = None

        rect = windows[index].getRect()

        save_crop(base_dir, image_name, index, crop, crop_mask, bands_img, rect, geotrans, geoproj,
                  raster_dtype, mask_dtype)

    if include_mask:
        create_train_test_split(base_dir, split=split)


# Load the JSON Params
def load_json_params(json_path):
    """
    Load parameters from a JSON file and extract the values.

    Parameters:
    -----------
    json_path: Path to the JSON file containing the parameters.

    Returns:
    --------
    params: A dictionary containing the parameters.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r') as json_file:
        params = json.load(json_file)

    return params