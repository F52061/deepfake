import os, sys
print("CUDA_VISIBLE_DEVICES env:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
import torch
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i}: {torch.cuda.get_device_properties(i).name}")
