import sys
import shutil
import os
import cv2
import numpy as np
import time

import image3

class Config:

    # Target size for 2d slices
    size_img = np.array((256, 256))

    # Window/Level parameters.
    # -1 stands for normalization after clipping brightest 1%
    windows = np.array((1000, -1, -1))
    levels = np.array((500, -1, -1))


class PathSettings:

    volumes_ending_img = ["_fat_fraction.vtk", "_fat.vtk", "_water.vtk"]


def main(argv):

    dataset_name = "slices_tellus"

    volume_list_path = "config/volume_list.txt"

    volumesToSlices(dataset_name, volume_list_path)


def volumesToSlices(dataset_name, volume_list_path):

    dataset_path = "data/" + dataset_name + "/"

    #
    print("########## Creating dataset {}".format(dataset_name))

    # Create output folder
    if os.path.exists(dataset_path):
        print("ABORT: Folder already exists!")
        sys.exit()

    os.makedirs(dataset_path)

    #
    start_time = time.time()

    path_settings = PathSettings()
    config = Config()

    # 
    with open(volume_list_path) as f:
        volume_names = f.read().splitlines()

    processVolumes(dataset_path, volume_names, config, path_settings)

    end_time = time.time()
    print("Elapsed time: {}".format(end_time - start_time))


def processVolumes(dataset_path, volume_names, config, path_settings):

    # Loop over images in set
    for i in range(len(volume_names)):

        volume_name = volume_names[i]

        #
        output_prefix = dataset_path + os.path.basename(os.path.normpath(volume_name))
            
        # Load and pre-process volumes and labels
        volume_img = loadVolumes(path_settings, volume_name, config)

        # Convert volumes to two-dimensional slices
        extractVolumeSlices(volume_img, config, output_prefix)


def extractVolumeSlices(volume_img, config, output_prefix):

    C = len(volume_img)
    dim = volume_img[0].data.shape

    print output_prefix

    # Loop over slices of image
    for z in range(dim[0]):

        # Extract slice
        slice_img = np.zeros((C, dim[1], dim[2])).astype(volume_img[0].data.dtype)
        for c in range(C):
            slice_img[c, :, :] = volume_img[c].data[z, :, :]

        # Format to target dimensions
        slice_img = formatSlice(slice_img, config)

        # Create three-digit index
        index = "00{}".format(z)
        index = index[len(index) - 3:len(index)]

        # Write slices
        cv2.imwrite(output_prefix + "_slice_{}_img.png".format(index), slice_img)


def loadVolumes(path_settings, volume_name, config):

    # Get paths to image volumes that form the channels of the 2d slice
    volume_path_img = []
    for i in range(len(path_settings.volumes_ending_img)):
        volume_path_img.append(volume_name + path_settings.volumes_ending_img[i])

    # Read files from vtk format
    C = len(volume_path_img)

    volume_img = []
    for c in range(C):
        volume_img.append(image3.create_itk_image(volume_path_img[c]))

    preprocessVolumes(config, volume_img)

    return volume_img


def preprocessVolumes(config, volume_img):

    for c in range(len(volume_img)):
        # Convert to float
        volume_img[c].data = volume_img[c].data.astype('float32')

        # Convert to range [0, 1]
        if config.levels[c] == -1 or config.windows[c] == -1:

            # Remove brightest 1% of voxels per slice and normalize
            dim = volume_img[c].data.shape

            for z in range(dim[0]):

                slice = volume_img[c].data[z, :, :]
                threshold = getThreshold(slice)

                # Cut off anything above threshold
                idx = slice[:] > threshold
                slice[idx] = threshold

                slice = slice[:] - np.amin(slice)
                if not np.amax(slice) == 0:
                    slice = slice[:] / np.amax(slice)

                volume_img[c].data[z, :, :] = slice
        else:

            # Perform windowing
            min_value = config.levels[c] - config.windows[c] / 2
            max_value = config.levels[c] + config.windows[c] / 2

            volume_img[c].data[:] = (volume_img[c].data[:] - min_value) / max_value

            idx_a = volume_img[c].data[:] < 0.0
            volume_img[c].data[idx_a] = 0.0

            idx_b = volume_img[c].data[:] > 1.0
            volume_img[c].data[idx_b] = 1.0

        # Convert from float in [0, 1] to 16bit
        volume_img[c].data[:] = volume_img[c].data[:] * 65535
        volume_img[c].data = volume_img[c].data.astype('uint16')


# Get intensity value for brightest 1% of pixels
def getThreshold(slice):

    sum = 0
    limit = 0.01 * slice.size

    bins = np.unique(slice[:])

    t = np.amax(bins)

    for i in range(len(bins)):
        t = bins[len(bins) - i - 1]

        sum += np.count_nonzero(slice == t)

        if sum >= limit:
            break

    return t


def formatSlice(slice, config):

    C = 3

    target_size = config.size_img

    if len(slice.shape) > 2:
        slice_shape = np.array((slice.shape[1], slice.shape[2]))
    else:
        slice_shape = np.array((slice.shape[0], slice.shape[1]))

    # Scale up to at least target size
    if slice_shape[0] < target_size[0] or slice_shape[1] < target_size[1]:

        # Pad image to target size
        grow = target_size - slice_shape

        # Add lower and upper padding, which may be asymmetric
        grow_l = grow / 2
        grow_u = np.ceil(grow / 2.0).astype('int')

        if len(slice.shape) > 2:
            slice = np.lib.pad(slice, 
                            (
                                (0, 0),
                                (grow_l[0], grow_u[0]),
                                (grow_l[1], grow_u[1])
                            ), 
                            "constant", constant_values=(0, 0))
        else:
            slice = np.lib.pad(slice, 
                            (
                                (grow_l[0], grow_u[0]),
                                (grow_l[1], grow_u[1])
                            ), 
                            "constant", constant_values=(0, 0))

    if len(slice.shape) > 2:
        slice_shape = np.array((slice.shape[1], slice.shape[2]))
    else:
        slice_shape = np.array((slice.shape[0], slice.shape[1]))

    # Crop to at most target size
    if slice_shape[0] > target_size[0] or slice_shape[1] > target_size[1]:

        # Crop image to target size
        centre = np.array(slice_shape) / 2

        read_start = centre - target_size / 2
        read_end = centre + np.ceil(target_size / 2.0).astype('int')
        
        slice = slice[:, read_start[0]:read_end[0], read_start[1]:read_end[1]]

    if len(slice.shape) > 2:
        slice = slice.swapaxes(0, 2)
        slice = slice.swapaxes(0, 1)

    return slice


if __name__ == '__main__':
    main(sys.argv)
