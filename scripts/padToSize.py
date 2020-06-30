import os
import sys
import numpy as np
import shutil

def main(argv):

    data_path = "data/slices_tellus/"

    # Number of desired slices
    target_size = 24

    padToSize(data_path, target_size)


def padToSize(data_path, target_size):

    print("##### Padding volumes to {} slices for 3D processing".format(target_size))

    #
    slices = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    slices_img = [f for f in slices if "img.png" in f]

    #
    N = len(slices_img)

    volume_names = []
    slice_indices = np.zeros(N)

    # Extract volume names and slice indices
    for i in range(N):

        #
        name_start = slices_img[i].find("/")
        name_end = slices_img[i].find("_slice_")

        volume_name = slices_img[i][name_start + 1:name_end]
        volume_names.append(volume_name)

        #
        slice_index = slices_img[i][name_end+7:name_end+10]
        slice_indices[i] = int(slice_index)

    #
    volume_names_unique = np.unique(volume_names)
    M = len(volume_names_unique)

    #
    for i in range(M):

        volume_name = volume_names_unique[i].strip()

        volume_slices_img = []
        volume_slice_indices = []

        aug_suffix = volume_name.find("warp")
        is_aug = (aug_suffix != -1)

        #
        for j in range(N):

            aug_suffix_j = slices_img[j].find("warp")
            is_aug_j = (aug_suffix_j != -1)

            #
            if volume_name in slices_img[j] and is_aug == is_aug_j:
                volume_slices_img.append(slices_img[j])
                volume_slice_indices.append(int(slice_indices[j]))
        
        #
        (volume_slice_indices, volume_slices_img) = zip(*sorted(zip(volume_slice_indices, volume_slices_img)))

        #
        slice_count = np.amax(volume_slice_indices) + 1

        print("Padding {} from {} to {} slices".format(volume_name, slice_count, target_size))

        slice_path_last = data_path + "/" + volume_slices_img[slice_count - 1]
        old_index = "00{}".format(slice_count - 1)
        old_index = old_index[len(old_index)-3:len(old_index)]

        #
        if slice_count > target_size:

            for j in range(target_size, slice_count):

                new_index = "00{}".format(j)
                new_index = new_index[len(new_index)-3:len(new_index)]

                slice_path = slice_path_last.replace("slice_{}".format(old_index), "slice_{}".format(new_index))

                os.remove(slice_path)

        #
        if slice_count < target_size:

            for j in range(slice_count, target_size):

                new_index = "00{}".format(j)
                new_index = new_index[len(new_index)-3:len(new_index)]

                slice_path_new = slice_path_last.replace("slice_{}".format(old_index), "slice_{}".format(new_index))

                shutil.copy(slice_path_last, slice_path_new) 


if __name__ == '__main__':
    main(sys.argv)
