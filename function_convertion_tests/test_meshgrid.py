import pickle
import numpy as np
import megengine as mge

import torch
import torch.nn.functional as F

def test_meshgrid():
    # Getting back the megengine objects:
    with open('test_data/meshgrid_np_test.pkl', 'rb') as f:
        rx, dilatex, ry, dilatey, x_grid, y_grid = pickle.load(f)

    x_grid = x_grid.numpy()
    y_grid = y_grid.numpy()

    # Test Pytorch
    x_grid_pytorch, y_grid_pytorch = torch.meshgrid(torch.arange(-rx, rx + 1, dilatex, device='cpu'), 
                                     torch.arange(-ry, ry + 1, dilatey, device='cpu'), indexing='xy')


    error_x = np.mean(x_grid_pytorch.numpy()-x_grid)
    error_y = np.mean(y_grid_pytorch.numpy()-y_grid)
    print(f"test_meshgrid (X) - Avg. Error: {error_x},  \n \
        Obtained shape: {x_grid_pytorch.numpy().shape}, Expected shape: {x_grid.shape}")
    print(f"test_meshgrid (Y) - Avg. Error: {error_y},  \n \
        Obtained shape: {y_grid_pytorch.numpy().shape}, Expected shape: {y_grid.shape}")

if __name__ == '__main__':

    test_meshgrid()