import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Result:
# 1
# NVIDIA GeForce GTX 1050