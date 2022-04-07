import pickle
import numpy as np
import megengine as mge

import torch
import torch.nn.functional as F

def test_offset():
    # Getting back the megengine objects:
    with open('test_data/offset_test.pkl', 'rb') as f:
        x_grid, y_grid, reshape_shape, transpose_order, expand_size, repeat_size, repeat_axis, offsets = pickle.load(f)

    x_grid = torch.tensor(x_grid.numpy())
    y_grid = torch.tensor(y_grid.numpy())
    offsets_mge = offsets.numpy()
    N = repeat_size

    # Test Pytorch
    offsets = torch.stack((x_grid, y_grid))
    offsets = offsets.reshape(2, -1).permute(1, 0)
    for d in sorted((0, 2, 3)):
        offsets = offsets.unsqueeze(d)
    offsets = offsets.repeat_interleave(N, dim=0)

    error = np.mean(offsets.numpy()-offsets_mge)
    print(f"test_offset - Avg. Error: {error},  \n \
        Obtained shape: {offsets.numpy().shape}, Expected shape: {offsets_mge.shape}")

if __name__ == '__main__':

    test_offset()