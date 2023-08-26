# CREStereo-Pytorch
 Non-official Pytorch implementation of the CREStereo (CVPR 2022 Oral) model converted from the original MegEngine implementation.

![!CREStereo-Pytorch stereo detph estimation](https://github.com/ibaiGorordo/CREStereo-Pytorch/blob/main/doc/img/output.jpg)

**update 2023/01/03**:
- enable DistributedDataParallel (DDP) training, training time is much faster than before.  

```shell
# train DDP
# change 'dist' to True in /cfgs/train.yaml file
python -m torch.distributed.launch --nproc_per_node=8 train.py
# train DP
# change 'dist' to False in /cfgs/train.yaml file
python train.py
```
 
# Important
- This is just an effort to try to implement the CREStereo model into Pytorch from MegEngine due to the issues of the framework to convert to other formats (https://github.com/megvii-research/CREStereo/issues/3).
- I am not the author of the paper, and I am don't fully understand what the model is doing. Therefore, there might be small differences with the original model that might impact the performance.
- I have not added any license, since the repository uses code from different repositories. Check the License section below for more detail.

# Pretrained model
- Download the model from [here](https://drive.google.com/file/d/1D2s1v4VhJlNz98FQpFxf_kBAKQVN_7xo/view?usp=sharing) and save it into the **[models](https://github.com/ibaiGorordo/CREStereo-Pytorch/tree/main/models)** folder.
- The model was converted from the original **[MegEngine weights](https://drive.google.com/file/d/1Wx_-zDQh7BUFBmN9im_26DFpnf3AkXj4/view)** using the `convert_weights.py` script. Place the MegEngine weights (crestereo_eth3d.mge) file into the **[models](https://github.com/ibaiGorordo/CREStereo-Pytorch/tree/main/models)** folder before the conversion.

# ONNX Conversion
- After either downloading the pretrained weights or training your own model, you will have a `models/crestereo_eth3d.pth` file. If you want to run your model with ONNX, you need to run the convert_to_onnx.py script. The script has two parts:
1. Convert the model to an ONNX model that takes in left, right images as well as an initial flow estimate (takes a few seconds)
2. Convert the model to an ONNX model that takes in left, right images and NO initial flow estimate (takes several minutes and requires pytorch >= 1.12)
(afaik) You will need both models to get the same results as you do from test_model.py.
- Run the test_onnx_model.py script to verify your models work as expected!
- NOTE: although the test_model.py script works with any size images as input, once you have converted your
Pytorch model into ONNX models, you must provide them with the image sizes used at conversion time or it will not work.

# Licences:
- CREStereo (Apache License 2.0): https://github.com/megvii-research/CREStereo/blob/master/LICENSE
- RAFT (BSD 3-Clause):https://github.com/princeton-vl/RAFT/blob/master/LICENSE
- LoFTR (Apache License 2.0):https://github.com/zju3dv/LoFTR/blob/master/LICENSE

# References:
- CREStereo: https://github.com/megvii-research/CREStereo
- RAFT: https://github.com/princeton-vl/RAFT
- LoFTR: https://github.com/zju3dv/LoFTR
- Grid sample replacement: https://zenn.dev/pinto0309/scraps/7d4032067d0160
- torch2mge: https://github.com/MegEngine/torch2mge
