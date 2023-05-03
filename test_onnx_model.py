import numpy as np
import cv2

import onnxruntime

# Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, no_flow_model):
    # Get onnx model layer names (see convert_to_onnx.py for what these are)
    input1_name = model.get_inputs()[0].name
    input2_name = model.get_inputs()[1].name
    input3_name = model.get_inputs()[2].name
    output_name = model.get_outputs()[0].name

    # Decimate the image to half the original size for flow estimation network
    imgL_dw2 = cv2.resize(
        left, (left.shape[1] // 2, left.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    imgR_dw2 = cv2.resize(
        right, (right.shape[1] // 2, right.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    # Reshape inputs to match what is expected
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :]).astype("float32")
    imgR = np.ascontiguousarray(imgR[None, :, :, :]).astype("float32")

    imgL_dw2 = imgL_dw2.transpose(2, 0, 1)
    imgR_dw2 = imgR_dw2.transpose(2, 0, 1)
    imgL_dw2 = np.ascontiguousarray(imgL_dw2[None, :, :, :]).astype("float32")
    imgR_dw2 = np.ascontiguousarray(imgR_dw2[None, :, :, :]).astype("float32")

    print("Model Forwarding...")
    # First pass it just to get the flow
    pred_flow_dw2 = no_flow_model.run(
        [output_name], {input1_name: imgL_dw2, input2_name: imgR_dw2})[0]
    # Second pass gets us the disparity
    pred_disp = model.run([output_name], {
                          input1_name: imgL, input2_name: imgR, input3_name: pred_flow_dw2})[0]

    return np.squeeze(pred_disp[:, 0, :, :])


if __name__ == '__main__':

    left_img = cv2.imread("left.png")
    right_img = cv2.imread("right.png")

    in_h, in_w = left_img.shape[:2]

    # Resize images
    eval_h, eval_w = (in_h, in_w)
    assert eval_h % 8 == 0, "input height should be divisible by 8"
    assert eval_w % 8 == 0, "input width should be divisible by 8"

    imgL = cv2.resize(left_img, (eval_w, eval_h),
                      interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right_img, (eval_w, eval_h),
                      interpolation=cv2.INTER_LINEAR)

    no_flow_model_path = "models/crestereo_without_flow.onnx"
    model_path = "models/crestereo.onnx"

    model = onnxruntime.InferenceSession(model_path)
    no_flow_model = onnxruntime.InferenceSession(no_flow_model_path)

    pred = inference(imgL, imgR, model, no_flow_model)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (eval_w, eval_h),
                      interpolation=cv2.INTER_LINEAR) * t
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    combined_img = np.hstack((left_img, disp_vis))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", combined_img)
    cv2.imwrite("output.jpg", disp_vis)
    cv2.waitKey(0)
