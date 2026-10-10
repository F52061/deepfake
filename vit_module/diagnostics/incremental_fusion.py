"""G30: compare ViT-only, global CLIP, and repaired Bridge increments."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch
from torch import nn

IDENTITY = ("row_id", "path", "domain", "split", "video_id", "y")
MODES = ("V_MATCHED", "V_CLIP", "V_BRIDGE")

class Head(nn.Module):
    def __init__(self, dims, seed):
        super().__init__(); torch.manual_seed(seed)
        self.net = nn.Linear(dims, 1)
    def forward(self, x): return self.net(x).squeeze(-1)

def load(root):
    files = sorted(Path(root).glob("clean_*.npz"))
    if not files: raise ValueError("Missing clean_*.npz")
    keys = list(IDENTITY) + ["V", "C", "B"]
    cols = {k: [] for k in keys}
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in keys if k not in z]
            if missing: raise ValueError(f"{path} missing {missing}")
            for k in keys: cols[k].append(z[k])
    data = {k: np.concatenate(v) for k, v in cols.items()}
    if not np.array_equal(data["row_id"], np.arange(len(data["y"]))): raise ValueError("row order invalid")
    return data

def groups(data, source):
    train = (data["domain"] == source) & (data["split"] == "train")
    if train.sum() < 2 or len(np.unique(data["y"][train])) != 2: raise ValueError("invalid source train")
    return train

def fit(x, y, train, decay, seed, epochs, lr):
    scaler = StandardScaler().fit(x[train]); xs = scaler.transform(x).astype("float32")
    model = Head(xs.shape[1], seed); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    xx, yy = torch.from_numpy(xs[train]), torch.from_numpy(y[train].astype("float32"))
    for _ in range(epochs):
        opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(model(xx), yy); loss.backward(); opt.step()
    with torch.no_grad(): score = model(torch.from_numpy(xs)).numpy()
    return score, {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "decay": decay}

def choose_decay(x, y, train_rows, video_ids, decays, seed, epochs, lr):
    folds = GroupKFold(n_splits=3).split(train_rows, y[train_rows], video_ids[train_rows])
    losses = {decay: [] for decay in decays}
    for fold, (fit_idx, val_idx) in enumerate(folds):
        fit_rows, val_rows = train_rows[fit_idx], train_rows[val_idx]
        for decay in decays:
            score, _ = fit(x, y, fit_rows, decay, seed + fold, epochs, lr)
            z = np.clip(score[val_rows], -40, 40)
            losses[decay].append(float(np.mean(np.logaddexp(0, z) - y[val_rows] * z)))
    return min(decays, key=lambda decay: (np.mean(losses[decay]), decay))

def auc(y, s, mask): return float(roc_auc_score(y[mask], s[mask])) if len(np.unique(y[mask])) == 2 else None

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--source", default="ffpp"); p.add_argument("--primary-domains", nargs="+", default=["cd2", "dfdcp", "wild"])
    p.add_argument("--seeds", nargs="+", type=int, default=[20261040,20261041,20261042]); p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=.001); p.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4,1e-3,1e-2]); p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--native-score-npz", default=None)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=False, exist_ok=False); d=load(a.input); train=groups(d,a.source)
    feats={"V_MATCHED":d["V"], "V_CLIP":np.c_[d["V"],d["C"]], "V_BRIDGE":np.c_[d["V"],d["C"],d["B"]]}
    test=d["split"]=="test"; scores={}; records=[]
    if a.native_score_npz:
        with np.load(a.native_score_npz, allow_pickle=False) as native:
            if not np.array_equal(native["row_id"], d["row_id"]): raise ValueError("native row_id mismatch")
            scores["V_NATIVE"] = native["score"].astype("float32")
            for domain in a.primary_domains:
                mask = test & (d["domain"] == domain)
                records.append({"key":"V_NATIVE", "domain":domain, "auc":auc(d["y"], scores["V_NATIVE"], mask), "source_auc":None, "decay":None})
    for seed in a.seeds:
        for mode,x in feats.items():
            decay = choose_decay(x, d["y"], np.flatnonzero(train), d["video_id"], a.weight_decays, seed+len(mode), a.epochs, a.lr)
            s,meta=fit(x,d["y"],train,decay,seed+len(mode),a.epochs,a.lr)
            best=(auc(d["y"],s,train),s,meta)
            key=f"{mode}|{seed}"; scores[key]=best[1]
            for domain in a.primary_domains:
                m=test&(d["domain"]==domain); records.append({"key":key,"domain":domain,"auc":auc(d["y"],best[1],m),"source_auc":best[0],"decay":best[2]["decay"]})
    np.savez_compressed(out/"scores.npz", **{k:v for k,v in scores.items()})
    (out/"summary.json").write_text(json.dumps({"records":records,"modes":MODES,"primary_domains":a.primary_domains},indent=2),encoding="utf-8")
    (out/"config.json").write_text(json.dumps(vars(a),indent=2),encoding="utf-8"); (out/"COMPLETE").write_text("completed\n",encoding="utf-8")
if __name__ == "__main__": main()
