import torch

from nets import Model

model = Model(max_disp=256, mixed_precision=False, test_mode=True)
model.eval()

t1 = torch.rand(1, 3, 480, 640)
t2 = torch.rand(1, 3, 480, 640)

output = model(t1,t2)
print(output.shape)

