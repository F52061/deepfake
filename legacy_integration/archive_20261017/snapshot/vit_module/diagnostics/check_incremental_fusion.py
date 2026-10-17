"""Small smoke check for the G30 feature-level runner."""
import tempfile, subprocess, sys
from pathlib import Path
import numpy as np
root=Path(tempfile.mkdtemp()); n=48; rng=np.random.default_rng(3)
kw={"row_id":np.arange(n),"path":np.array([str(i) for i in range(n)]),"domain":np.array(["ffpp"]*24+["cd2"]*24),"split":np.array(["train"]*16+["test"]*32),"video_id":np.array([f"v{i//2}" for i in range(n)]),"y":np.array([0,1]*(n//2)),"V":rng.normal(size=(n,8)).astype("float32"),"C":rng.normal(size=(n,5)).astype("float32"),"B":rng.normal(size=(n,4)).astype("float32")}
np.savez(root/"clean_00000000.npz",**kw); out=root/"out"
subprocess.run([sys.executable,str(Path(__file__).with_name("incremental_fusion.py")),"--input",str(root),"--output",str(out),"--epochs","2","--bootstrap","0"],check=True)
assert (out/"COMPLETE").exists() and (out/"scores.npz").exists(); print("PASS: G30 feature loading, three heads, output archive")
