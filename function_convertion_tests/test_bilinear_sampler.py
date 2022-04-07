import pickle
import numpy as np
import megengine as mge

import torch
import torch.nn.functional as F

def bilinear_sampler(img, coords, mode='bilinear', mask=False):

    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def test_bilinear_sampler():
    # Getting back the megengine objects:
    with open('test_data/bilinear_sampler_test.pickle', 'rb') as f:
        right_feature_prev, coords, right_feature = pickle.load(f)

    right_feature_prev = torch.tensor(right_feature_prev.numpy())
    coords = torch.tensor(coords.numpy())
    right_feature = right_feature.numpy()

    # Test Pytorch
    right_feature_pytorch = bilinear_sampler(right_feature_prev, coords).numpy()

    error = np.mean(right_feature_pytorch-right_feature)
    print(f"test_coords_grid - Avg. Error: {error},  \n \
        Original shape: {coords.numpy().shape},\n  \
        Obtained shape: {right_feature_pytorch.shape}, Expected shape: {right_feature.shape}")

if __name__ == '__main__':

    test_bilinear_sampler()