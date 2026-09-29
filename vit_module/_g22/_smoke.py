import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g22_common as C
ok, info = C.gate_on_gpu()
print('[gpu]', info, 'OK=', ok)
C.pin_gpu(1)
import torch
print('cuda visible:', torch.cuda.device_count(), torch.cuda.get_device_name(0))
m = C.build_model(device='cuda:0', verbose=True)
print('model built')
print('bridge_adapter_proj.bridge_adapter_proj =', m.bridge_adapter_proj.bridge_adapter_proj)
print('mem MiB:', torch.cuda.memory_allocated()/2**20)
