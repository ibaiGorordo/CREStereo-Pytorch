import torch
import torch.nn.functional as F
import numpy as np
import cv2
from imread_from_url import imread_from_url
from nets.utils.utils import dxs,dys

from nets import Model

device = 'cuda'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, size, n_iter=20):

    print("Model Forwarding...")
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = torch.tensor(imgL.astype("float32")).to(device)
    imgR = torch.tensor(imgR.astype("float32")).to(device)

    imgL_dw2 = F.interpolate(
        imgL,
        size=size,
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.interpolate(
        imgR,
        size=size,
        mode="bilinear",
        align_corners=True,
    )
    # print(imgR_dw2.shape)
    with torch.inference_mode():
        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

        pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

    return pred_disp

if __name__ == '__main__':

    left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
    right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")

    in_h, in_w = left_img.shape[:2]

    # Resize image in case the GPU memory overflows
    eval_h, eval_w = (in_h,in_w)
    assert eval_h%8 == 0, "input height should be divisible by 8"
    assert eval_w%8 == 0, "input width should be divisible by 8"

    imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    model_path = "models/crestereo_eth3d.pth"
    print(dxs,dys)

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.to(device)
    model.eval()

    scale_factor,downsample = 1,True    # max(dx),max(dy)=21,4
    scale_factor,downsample = 2,True    # max(dx),max(dy)=22,4
    scale_factor,downsample = 4,True    # max(dx),max(dy)=27,4
    scale_factor,downsample = 2,False   # max(dx),max(dy)=32,4
    scale_factor,downsample = 4,False   # max(dx),max(dy)=46,5

    assert imgL.shape == imgR.shape
    H,W,_ = imgL.shape
    def resize(scale_factor, downsample): # {
        assert scale_factor in [1,2,4]
        if downsample:
            H_, W_ = H // scale_factor, W // scale_factor
        else:
            H_, W_ = H * scale_factor, W * scale_factor
        print(f'H*W={H*W}')
        print(f'H_*W_={H_*W_}')
        return (H_, W_)
    # }; end of resize
    size = (H_,W_) = resize(scale_factor, downsample)
    pred = inference(imgL, imgR, model, size, n_iter=20)
#   print(dxs, dys)
    print(max(dxs), max(dys))

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    combined_img = np.hstack((left_img, disp_vis))
#   cv2.namedWindow("output", cv2.WINDOW_NORMAL)
#   cv2.imshow("output", combined_img)
    _ = '_' if downsample else ''
    output_file = f'output{H}x{W}-{H_}x{W_}.jpg'
    cv2.imwrite(output_file, disp_vis)
    cv2.waitKey(0)



