import torch
import torch.nn.functional as F

from .utils import bilinear_sampler, coords_grid

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

    def _shifts(self, warped, small_patch):
        """3x3 (small_patch) or 1x9 neighbourhood of the warped right features: list of 9 shifted views of the
        replicate-padded map, in upstream's order (row-major)."""
        H, W = warped.shape[-2:]
        pady, padx = (1, 1) if small_patch else (0, 4)
        wp = F.pad(warped, (padx, padx, pady, pady), "replicate")
        return [
            wp[:, :, h : h + H, w : w + W] for h in range(2 * pady + 1) for w in range(2 * padx + 1)
        ]

    @staticmethod
    def _group_corr(left, rights, groups=4):
        """mean over each channel group of left * right, for each right in rights -> [N, groups * len(rights), H, W]
        (channel = group * len(rights) + index, as upstream's per-group concatenation)."""
        N, C, H, W = left.shape
        lg = left.reshape(N, groups, C // groups, H, W)
        out = [(lg * r.reshape(N, groups, C // groups, H, W)).mean(2) for r in rights]
        return torch.stack(out, 2).reshape(N, groups * len(rights), H, W)

    def corr_iter(self, left_feature, right_feature, flow, small_patch):
        # the 4 channel groups share the warp and the search window: warp once, correlate all groups per shift
        coords = (self.coords + flow).permute(0, 2, 3, 1)
        warped = bilinear_sampler(right_feature, coords)
        return self._group_corr(left_feature, self._shifts(warped, small_patch))

    def _attended(self, left_feature, right_feature):
        """cross attention of the 1/16 features: its inputs are the same in every iteration -> computed once."""
        if getattr(self, "_att_cache", None) is None:
            N, C, H, W = left_feature.shape
            left_tokens = left_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)  # 'n c h w -> n (h w) c'
            right_tokens = right_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)
            left_tokens, right_tokens = self.att(left_tokens, right_tokens)
            self._att_cache = [
                x.reshape(N, H, W, C).permute(0, 3, 1, 2) for x in (left_tokens, right_tokens)
            ]  # 'n (h w) c -> n c h w'
        return self._att_cache

    def corr_att_offset(
        self, left_feature, right_feature, flow, extra_offset, small_patch
    ):

        N, C, H, W = left_feature.shape

        if self.att is not None:
            left_feature, right_feature = self._attended(left_feature, right_feature)

        if small_patch:
            ry, rx = 1, 1
        else:
            ry, rx = 0, 4
        search_num = 9
        extra_offset = extra_offset.reshape(N, search_num, 2, H, W).permute(0, 1, 3, 4, 2)  # [N, search_num, H, W, 2]
        x_grid, y_grid = torch.meshgrid(torch.arange(-rx, rx + 1, device=left_feature.device),
                                        torch.arange(-ry, ry + 1, device=left_feature.device), indexing='xy')
        offsets = torch.stack((x_grid, y_grid)).reshape(2, -1).permute(1, 0).reshape(1, search_num, 1, 1, 2)
        offsets = offsets + extra_offset

        coords = (self.coords + flow).permute(0, 2, 3, 1)  # [N, H, W, 2]
        coords = (coords.unsqueeze(1) + offsets).reshape(N, -1, W, 2)  # [N, search_num*H, W, 2]

        # the 4 channel groups use identical sampling positions: sample all channels at once
        right = bilinear_sampler(right_feature, coords).reshape(N, C, search_num, H, W)
        g = 4
        corr = (left_feature.reshape(N, g, C // g, 1, H, W) * right.reshape(N, g, C // g, search_num, H, W)).mean(2)
        return corr.reshape(N, g * search_num, H, W)
