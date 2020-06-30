import sys
import shutil
import os
import cv2
import numpy as np
import time
import random
import copy
import torch.nn.functional as F

sys.path.insert(0, './scripts/v_net')
from model_vnet import VNet

import torch
from torch.autograd import Variable
import torch.nn as nn

def processSlices(data_path, network_path, output_path):

    if os.path.exists(output_path):
        print("ABORT: Output folder already exists!")
        sys.exit()

    print("########## Applying V-Net")

    os.makedirs(output_path)

    #
    start_time = time.time()

    (X, slice_names) = loadData(data_path)

    net = initializeNetwork(network_path)
    applyNetwork(net, X, output_path, slice_names)

    end_time = time.time()
    print("Elapsed time: {}".format(end_time - start_time))


def loadData(data_path):

    slices = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    slices = [f for f in slices if "img" in f]

    slices = sorted(slices)

    C = 3
    x_size = 256
    y_size = 256
    z_size = 24

    N = len(slices)
    M = N / z_size

    # Convert volume to tensor filled with 3d volumes
    X = Variable(torch.zeros(M, C, z_size, x_size, y_size))

    z = 0
    v = 0

    for i in range(N):

        img = cv2.imread(data_path + "/" + slices[i], -2)
        img = img.astype(np.float32) / (256.*256.) # normalize from 16bit

        for c in range(C):
            X[v, c, z, :, :] = torch.from_numpy(img[:, :, c])

        # Get index to next slice
        z = (z + 1)

        # Switch to next volume if all slices of previous one have been loaded
        if z >= z_size:
            v = v + 1
            z = 0

    return (X, slices)


def initializeNetwork(network_path):

    # Initialize network
    net = VNet().cuda()

    snapshot = torch.load(network_path)
    net.load_state_dict(snapshot['state_dict'])

    net.eval()
    torch.backends.cudnn.benchmark = True

    return net


def applyNetwork(net, X, output_path, slice_names):

    dim = X.size()

    N = dim[0]
    C = dim[1]
    z_size = dim[2]
    y_size = dim[3]
    x_size = dim[4]

    # Slice name index
    i = 0

    # Predict volume-wise
    for v in range(N):
        
        print("Processing volume {} of {}...".format(v+1, N))

        output = net(X[v:v+1, :, :, :, :].cuda())

        #
        out = np.argmax(output.cpu().data.numpy(), 1)
        seg = out.reshape((24, 256, 256)) * 127

        for z in range(z_size):

            out_path = output_path + slice_names[i]

            cv2.imwrite(out_path, seg[z, :, :])

            i += 1
