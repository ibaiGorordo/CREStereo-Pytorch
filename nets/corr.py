import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import bilinear_sampler, coords_grid, manual_pad

class AGCL:
    """
    Implementation of Adaptive Group Correlation Layer (AGCL).
    """

    def __init__(self, fmap1, fmap2, att=None):
        self.fmap1 = fmap1
        self.fmap2 = fmap2

        self.att = att

        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)

    def __call__(self, flow, extra_offset, small_patch=False, iter_mode=False):
        if iter_mode:
            corr = self.corr_iter(self.fmap1, self.fmap2, flow, small_patch)
        else:
            corr = self.corr_att_offset(
                self.fmap1, self.fmap2, flow, extra_offset, small_patch
            )
        return corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.shape

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        right_pad = manual_pad(right_feature, pady, padx)

        corr_list = []
        for h in range(0, pady * 2 + 1, di_y):
            for w in range(0, padx * 2 + 1, di_x):
                right_crop = right_pad[:, :, h : h + H, w : w + W]
                assert right_crop.shape == left_feature.shape
                corr = torch.mean(left_feature * right_crop, dim=1, keepdims=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final

    def corr_iter(self, left_feature, right_feature, flow, small_patch):

        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature = bilinear_sampler(right_feature, coords)

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        N, C, H, W = left_feature.shape
        lefts = torch.split(left_feature, left_feature.shape[1]//4, dim=1)
        rights = torch.split(right_feature, right_feature.shape[1]//4, dim=1)

        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(
                lefts[i], rights[i], psize_list[i], dilate_list[i]
            )
            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr

    def corr_att_offset(
        self, left_feature, right_feature, flow, extra_offset, small_patch
    ):

        N, C, H, W = left_feature.shape

        if self.att is not None:
            left_feature = left_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)  # 'n c h w -> n (h w) c'
            right_feature = right_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)  # 'n c h w -> n (h w) c'
            # 'n (h w) c -> n c h w'
            left_feature, right_feature = self.att(left_feature, right_feature)
            # 'n (h w) c -> n c h w'
            left_feature, right_feature = [
                x.reshape(N, H, W, C).permute(0, 3, 1, 2)
                for x in [left_feature, right_feature]
            ]

        lefts = torch.split(left_feature, left_feature.shape[1]//4, dim=1)
        rights = torch.split(right_feature, right_feature.shape[1]//4, dim=1)

        C = C // 4

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        search_num = 9
        extra_offset = extra_offset.reshape(N, search_num, 2, H, W).permute(0, 1, 3, 4, 2) # [N, search_num, 1, 1, 2]

        corrs = []
        for i in range(len(psize_list)):
            left_feature, right_feature = lefts[i], rights[i]
            psize, dilate = psize_list[i], dilate_list[i]

            psizey, psizex = psize[0], psize[1]
            dilatey, dilatex = dilate[0], dilate[1]

            ry = psizey // 2 * dilatey
            rx = psizex // 2 * dilatex
            x_grid, y_grid = torch.meshgrid(torch.arange(-rx, rx + 1, dilatex, device=self.fmap1.device), 
                                    torch.arange(-ry, ry + 1, dilatey, device=self.fmap1.device), indexing='xy')

            offsets = torch.stack((x_grid, y_grid))
            offsets = offsets.reshape(2, -1).permute(1, 0)
            for d in sorted((0, 2, 3)):
                offsets = offsets.unsqueeze(d)
            offsets = offsets.repeat_interleave(N, dim=0)
            offsets = offsets + extra_offset

            coords = self.coords + flow  # [N, 2, H, W]
            coords = coords.permute(0, 2, 3, 1)  # [N, H, W, 2]
            coords = torch.unsqueeze(coords, 1) + offsets
            coords = coords.reshape(N, -1, W, 2)  # [N, search_num*H, W, 2]

            right_feature = bilinear_sampler(
                right_feature, coords
            )  # [N, C, search_num*H, W]
            right_feature = right_feature.reshape(N, C, -1, H, W)  # [N, C, search_num, H, W]
            left_feature = left_feature.unsqueeze(2).repeat_interleave(right_feature.shape[2], dim=2)

            corr = torch.mean(left_feature * right_feature, dim=1)

            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr
