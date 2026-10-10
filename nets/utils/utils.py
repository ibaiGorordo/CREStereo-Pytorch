import torch
import torch.nn.functional as F

#Ref: https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # F.grid_sample: same values as the previous gather implementation (zero padding, align_corners), one fused kernel
    # instead of 4 gathers with [N, C, H*W] int64 index tensors; exports as ONNX GridSample (opset >= 16, TensorRT)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img
    
def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def manual_pad(x, pady, padx):

    pad = (padx, padx, pady, pady)  
    return F.pad(x.clone().detach(), pad, "replicate")
