import pickle
import numpy as np
import megengine as mge

import torch
import torch.nn.functional as F

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def test_coords_grid():
    # Getting back the megengine objects:
    with open('test_data/coords_grid_test.pickle', 'rb') as f:
        batch, ht, wd, coords = pickle.load(f)

    coords = coords.numpy()

    # Test Pytorch
    coords_pytorch = coords_grid(batch, ht, wd, 'cpu').numpy()

    error = np.mean(coords_pytorch-coords)
    print(f"test_coords_grid - Avg. Error: {error},  \n \
        Obtained shape: {coords_pytorch.shape}, Expected shape: {coords.shape}")

if __name__ == '__main__':

    test_coords_grid()