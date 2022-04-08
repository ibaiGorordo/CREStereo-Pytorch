import copy
import torch
import numpy as np
import megengine as mge

from nets import Model

# Read Megengine parameters
pretrained_dict = mge.load("models/crestereo_eth3d.mge")

model = Model(max_disp=256, mixed_precision=False, test_mode=True)
model.eval()

state_dict = model.state_dict()
for key, value in pretrained_dict['state_dict'].items():

	print(f"Converting {key}")
	# Fix shape mismatch
	if value.shape[0] == 1:
		value = np.squeeze(value)

	state_dict[key] = torch.tensor(value)

output_path = "models/crestereo_eth3d.pth"
torch.save(state_dict, output_path)
print(f"\nModel saved to: {output_path}")