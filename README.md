# Fully Convolutional Networks for Automated Segmentation of Abdominal Adipose Tissue Depots in Multicenter Water-Fat MRI

This repository contains a PyTorch implementation of the segmentation method described in https://arxiv.org/abs/1807.03122v2.

It contains the trained weights for the U-Net and V-Net, which can be used to segment volumes in .vtk format using PyTorch.

Please note that we can not publically distribute the medical data used in the paper. If you wish to get access to reproduce the results, please contact us via the mail address given below. Unfortunately it may take weeks or months until access can be granted by the owners of the studies.

# Usage

Make sure to clone the repository in order to get the complete network snapshots (otherwise you may get an UnplickingError).

If you have image volumes in .vtk format, you can create input slices to the networks in the following way: Set the paths to your volumes in config/volume\_list.txt, leaving out the ".vtk" ending in the names. The volumes are expected to have dimensions of [256, 256, 21] and suffixes as listed in "PathSettings" of scripts/volumesToSlices.py. The segmentation volumes should be binary labels.

Next, execute run.py, which will run the chosen network ("unet" by default) on the volumes. The script will extract two-dimensional  slices, pre-process them according to the description in the paper, and generate output segmentations for them.

For any questions and feedback, feel free to contact taro.langner(at).surgsci.uu.se
