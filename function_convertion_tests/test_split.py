import pickle
import numpy as np
import megengine as mge

import torch
import torch.nn.functional as F

def test_split():
    # Getting back the megengine objects:
    with open('test_data/split_test.pkl', 'rb') as f:
        left_feature, size, axis, lefts = pickle.load(f)

    left_feature = torch.tensor(left_feature.numpy())

    # Test Pytorch
    lefts_pytorch = torch.split(left_feature, left_feature.shape[axis]//size, dim=axis)
    
    for i, (left_pytorch, left) in enumerate(zip(lefts_pytorch, lefts)):

        error = np.mean(left_pytorch.numpy()-left.numpy())
        print(f"test_split {i} - Avg. Error: {error},  \n \
            Obtained shape: {left_pytorch.numpy().shape}, Expected shape: {left.numpy().shape}\n")

def test_split_list():
    # Getting back the megengine objects:
    with open('test_data/split_test_list.pkl', 'rb') as f:
        fmap1, size, axis, net, inp = pickle.load(f)

    fmap1 = torch.tensor(fmap1.numpy())
    net = net.numpy()
    inp = inp.numpy()

    # Test Pytorch
    net_pytorch, inp_pytorch = torch.split(fmap1, [size[0],size[0]], dim=axis)

    error_net = np.mean(net_pytorch.numpy()-net)
    error_inp = np.mean(inp_pytorch.numpy()-inp)
    print(f"test_split_list (net) - Avg. Error: {error_net},  \n \
        Obtained shape: {net_pytorch.numpy().shape}, Expected shape: {net.shape}\n")
    print(f"test_split_list (inp) - Avg. Error: {error_inp},  \n \
        Obtained shape: {inp_pytorch.numpy().shape}, Expected shape: {inp.shape}\n")


if __name__ == '__main__':

    test_split()
    test_split_list()