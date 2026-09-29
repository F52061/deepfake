import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g22_common as C
ok, info = C.gate_on_gpu(1, 100); print('[gpu]', info, 'OK=', ok)
assert ok
C.pin_gpu(1)
import torch, torch.nn.functional as F
m = C.build_model(device='cuda:0', verbose=False)
_, IMT, _, _ = C.get_pipeline()
raw=[l.strip().split() for l in open(C.TRAIN_TXT,'r',encoding='utf-8') if l.strip()]
raw=[(p,int(l)) for p,l in raw if os.path.exists(p)]
N=64
t=time.time(); allimgs = torch.stack([C.load_image_tensor(p, IMT) for p,_ in raw[:N]]).to('cuda:0'); tl=time.time()-t
m.train()
for BS in (5, 6, 16, 32):
    with torch.no_grad(): _=m(allimgs[:BS])
    torch.cuda.synchronize()
    t=time.time()
    nb=0
    for i in range(0, N, BS):
        x=allimgs[i:i+BS]
        m.zero_grad(set_to_none=True)
        o=m(x); F.cross_entropy(o, torch.zeros(len(x),dtype=torch.long,device='cuda:0')).backward()
        nb+=1
    torch.cuda.synchronize()
    dt=time.time()-t
    print(f'BS={BS:3d}: {nb} steps, {dt:.2f}s for {N} imgs -> {dt/N*1000:.1f} ms/img -> {dt/N*107700/3600:.2f} h/epoch (compute only)')
print(f'load-only: {tl/N*1000:.1f} ms/img -> +{tl/N*107700/3600:.2f} h')
print('peak', torch.cuda.max_memory_allocated()/2**20)
