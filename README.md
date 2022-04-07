# CREStereo-Pytorch
 Non-official Pytorch implementation of the CREStereo(CVPR 2022 Oral).
 
# Important
- This is just an effort to try to implement the CREStereo model into Pytorch from MegEngine due to the issues of the framework to convert to other formats (https://github.com/megvii-research/CREStereo/issues/3).
- I am not the author of the paper, and I am don't fully understand what the model is doing. Therefore, there might be small differences with the original model that might impact the performance.
- The model has not been fully tested, I have tested parts of the model, but since I don't have the weight in Pytorch, I cannot test the actual output.
- Any help (checking errors, test to train the model, suggestions...) will be gratly appreciated.
- I have not added any license, since the repository uses code from different repositories. Check the License section below for more detail.

# Licences:
- CREStereo (Apache License 2.0): https://github.com/megvii-research/CREStereo/blob/master/LICENSE
- RAFT (BSD 3-Clause):https://github.com/princeton-vl/RAFT/blob/master/LICENSE
- LoFTR (Apache License 2.0):https://github.com/zju3dv/LoFTR/blob/master/LICENSE

# References:
- CREStereo: https://github.com/megvii-research/CREStereo
- RAFT: https://github.com/princeton-vl/RAFT
- LoFTR: https://github.com/zju3dv/LoFTR
- torch2mge: https://github.com/MegEngine/torch2mge
