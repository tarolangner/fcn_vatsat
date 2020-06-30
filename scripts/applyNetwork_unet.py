import sys
import shutil
import os
import cv2
import numpy as np
import time
import random
import copy
import torch.nn.functional as F

sys.path.insert(0, './scripts/u_net')
from model_unet import UNet

import torch
from torch.autograd import Variable
import torch.nn as nn

def processSlices(data_path, network_path, output_path):

    if os.path.exists(output_path):
        print("ABORT: Output folder already exists!")
        sys.exit()

    print("########## Applying U-Net")
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

    N = len(slices)
    C = 3
    x_size = 256
    y_size = 256

    # Convert volume to tensor filled with axial 2d slices
    X = Variable(torch.zeros(N, C, x_size, y_size))

    for i in range(N):

        img = cv2.imread(data_path + "/" + slices[i], -2)
        img = img.astype(np.float32) / (256.*256.) # normalize from 16bit

        for c in range(C):
            X[i, c, :, :] = torch.from_numpy(img[:, :, c])

    return (X, slices)


def initializeNetwork(network_path):

    # Initialize network
    net = UNet(channel_count = 3, class_count = 3).cuda()

    snapshot = torch.load(network_path)
    net.load_state_dict(snapshot['state_dict'])

    net.eval()
    torch.backends.cudnn.benchmark = True

    return net


def applyNetwork(net, X, output_path, slice_names):

    # Predict slice_wise
    N = len(slice_names)

    for i in range(N):

        print("Processing slice {} of {}...".format(i+1, N))

        output = net(X[i:i+1, :, :, :].cuda())
        output = F.log_softmax(output, dim=1)

        # Softmax probability
        m = nn.Softmax2d()
        output = m(output)

        prob_out = output.cpu().data[0, :, :, :].numpy()
        seg = np.argmax(prob_out, 0) * 127

        out_path = output_path + slice_names[i]

        cv2.imwrite(out_path, seg)
