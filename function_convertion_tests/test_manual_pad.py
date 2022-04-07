import pickle
import numpy as np
import megengine as mge

import torch
import torch.nn.functional as F

def manual_pad(x, pady, padx):

    pad = (padx, padx, pady, pady)  
    return F.pad(torch.tensor(x), pad, "replicate")


def test_pad_1_1():
    # Getting back the megengine objects:
    with open('test_data/manual_pad_test1_1.pickle', 'rb') as f:
        right_feature, pady, padx, right_pad = pickle.load(f)

    right_feature = right_feature.numpy()
    right_pad = right_pad.numpy()

    # Test Pytorch
    right_pad_pytorch = manual_pad(right_feature, pady, padx).numpy()

    error = np.mean(right_pad_pytorch-right_pad)
    print(f"test_pad_1_1 - Avg. Error: {error},  \n \
        Orig. shape: {right_feature.shape}, \n \
        Padded shape: {right_pad_pytorch.shape}, Expected shape: {right_pad.shape}")

def test_pad_0_4():
    # Getting back the megengine objects:
    with open('test_data/manual_pad_test0_4.pickle', 'rb') as f:
        right_feature, pady, padx, right_pad = pickle.load(f)

    right_feature = right_feature.numpy()
    right_pad = right_pad.numpy()

    # Test Pytorch
    right_pad_pytorch = manual_pad(right_feature, pady, padx).numpy()

    error = np.mean(right_pad_pytorch-right_pad)
    print(f"test_pad_0_4 - Avg. Error: {error},  \n \
        Orig. shape: {right_feature.shape}, \n \
        Padded shape: {right_pad_pytorch.shape}, Expected shape: {right_pad.shape}")


if __name__ == '__main__':

    test_pad_1_1()

    test_pad_0_4()