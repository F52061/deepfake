"""G8 共享低级图像统计套件（H4 浅层捷径归因用）。

两个 G8 子 agent 必须 import 本模块、调用 compute_stats()，保证所有统计定义一致。
仅依赖 numpy + PIL（scipy 有则用，无则纯 numpy 回退）。
所有统计在 224x224 LANCZOS 重采样图上计算（跨域尺度混淆最小化）；
native 分辨率单独记录，供审计。

统计字典 keys（全部 float）:
  h, w                  native 分辨率
  g_mean g_std          灰度均值/全局对比度
  sat_mean sat_std      HSV 饱和度 均值/标准差
  colorfulness          Hasler-Suessstrunk 色彩丰富度
  ch_diff               三通道均值极差 (gray-world violation)
  lapl_var              Laplacian 方差 (锐度)
  grad_mean             梯度幅值均值 (sobel 近似)
  edge_density          梯度>25 像素占比
  hi_en                 径向频率 >0.25*nyquist 能量占比 (高频能量)
  spec_slope            径向功率谱 log-log 斜率 (纹理粗细; 越负越粗)
  g_entropy             灰度 64 箱直方图熵
  noise_est             去高斯模糊后残差 std (高频噪声估计)
  blockiness            8px 网格边界不连续超额 (JPEG/网格痕迹, 原图才有意义, 这里在224图上)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"PIL unavailable: {e}")

try:
    import scipy.ndimage as ndi
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _grad(gray):
    """返回 (mag, gx, gy) —— 尽力纯 numpy 以实现, scipy 可用则用 sobel。"""
    if _HAVE_SCIPY:
        gx = ndi.sobel(gray, axis=1, mode="reflect")
        gy = ndi.sobel(gray, axis=0, mode="reflect")
    else:
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.hypot(gx, gy)
    return mag, gx, gy


def _laplacian_var(gray):
    if _HAVE_SCIPY:
        return float(ndi.laplace(gray, mode="reflect").var())
    # 纯 numpy: lap = up+down+left+right - 4*center
    u = np.zeros_like(gray); u[:-1, :] = gray[1:, :]
    dd = np.zeros_like(gray); dd[1:, :] = gray[:-1, :]
    l = np.zeros_like(gray); l[:, :-1] = gray[:, 1:]
    rr = np.zeros_like(gray); rr[:, 1:] = gray[:, :-1]
    lap = u + dd + l + rr - 4.0 * gray
    return float(lap.var())


def _residual_noise_std(gray):
    if _HAVE_SCIPY:
        blur = ndi.gaussian_filter(gray, 1.0, mode="reflect")
    else:
        # 纯 numpy 3x3 均值 (5 点交叉)
        u = np.zeros_like(gray); u[:-1, :] = gray[1:, :]
        dd = np.zeros_like(gray); dd[1:, :] = gray[:-1, :]
        l = np.zeros_like(gray); l[:, :-1] = gray[:, 1:]
        rr = np.zeros_like(gray); rr[:, 1:] = gray[:, :-1]
        blur = (gray + u + dd + l + rr) / 5.0
    return float((gray - blur).std())


def _spectral(gray):
    F = np.fft.fft2(gray)
    P = np.abs(F) ** 2
    fy = np.fft.fftfreq(gray.shape[0])
    fx = np.fft.fftfreq(gray.shape[1])
    r = np.hypot.outer(fy, fx)
    nyq = np.sqrt(0.5)  # corner nyquist (fx,fy each max 0.5)
    rmax = 0.5
    mask = (r > 0) & (r <= rmax)
    ps = P[mask]
    rr = r[mask]
    tot = ps.sum()
    hi = ps[rr > 0.25 * rmax].sum()
    hi_en = float(hi / tot) if tot > 0 else 0.0
    # radial power spectrum slope: log(P) ~ slope*log(r)
    edges = np.linspace(np.log(0.02), np.log(rmax), 24)
    idx = np.clip(np.searchsorted(edges, np.log(rr)) - 1, 0, len(edges) - 2)
    xb = np.empty(len(edges) - 1); yb = np.empty(len(edges) - 1)
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.sum() > 0:
            xb[k] = np.log(rr[sel]).mean()
            yb[k] = np.log(ps[sel] + 1e-12).mean()
        else:
            xb[k] = yb[k] = np.nan
    ok = ~np.isnan(xb) & ~np.isnan(yb)
    if ok.sum() >= 5:
        slope = float(np.polyfit(xb[ok], yb[ok], 1)[0])
    else:
        slope = float("nan")
    return hi_en, slope


def _blockiness(gray):
    """8px 边界垂直不连续超额比。正 = 网格伪影增强。"""
    d = np.abs(np.diff(gray, axis=1))  # (H, W-1)
    bc = np.arange(7, gray.shape[1] - 1, 8)  # 每 8px 边界列 (0-index 列 c 到 c+1)
    inter = np.ones(gray.shape[1] - 1, dtype=bool)
    inter[bc] = False
    mb = float(d[:, bc].mean()) if bc.size else float("nan")
    mi = float(d[:, inter].mean())
    if mi > 1e-9 and not np.isnan(mb):
        return float(mb / mi) - 1.0
    return float("nan")


def compute_stats(path, size=224):
    """对单张图返回统计 dict。文件读失败抛异常由调用方捕获跳过。"""
    img = Image.open(path).convert("RGB")
    W, H = img.size
    img = img.resize((size, size), Image.LANCZOS)
    a = np.asarray(img).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1).astype(np.float32)
    mn = a.min(-1).astype(np.float32)
    denom = mx + 1e-6
    sat = (mx - mn) / denom
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    mag, _, _ = _grad(gray)

    hi_en, spec_slope = _spectral(gray)

    hist, _ = np.histogram(gray, bins=64, range=(0.0, 255.0))
    p = hist / hist.sum()
    p = p[p > 0]
    g_entropy = float(-(p * np.log2(p)).sum())

    st = {
        "h": float(H), "w": float(W),
        "g_mean": float(gray.mean()), "g_std": float(gray.std()),
        "sat_mean": float(sat.mean()), "sat_std": float(sat.std()),
        "colorfulness": colorfulness,
        "ch_diff": float(max(r.mean(), g.mean(), b.mean()) - min(r.mean(), g.mean(), b.mean())),
        "lapl_var": _laplacian_var(gray),
        "grad_mean": float(mag.mean()),
        "edge_density": float((mag > 25.0).mean()),
        "hi_en": hi_en,
        "spec_slope": spec_slope,
        "g_entropy": g_entropy,
        "noise_est": _residual_noise_std(gray),
        "blockiness": _blockiness(gray),
    }
    return st


STAT_KEYS = [
    "g_mean", "g_std", "sat_mean", "sat_std", "colorfulness", "ch_diff",
    "lapl_var", "grad_mean", "edge_density", "hi_en", "spec_slope",
    "g_entropy", "noise_est", "blockiness",
]


def stats_array(paths, size=224, verbose=True):
    """批量算统计, 返回 (M, K) float 数组 + keys + 失败路径。统计失败样本以 NaN 行 + 记入 failed。"""
    keys = STAT_KEYS
    out = np.full((len(paths), len(keys)), np.nan, dtype=np.float64)
    failed = []
    for i, p in enumerate(paths):
        try:
            st = compute_stats(p, size=size)
            for j, k in enumerate(keys):
                out[i, j] = st[k]
        except Exception as e:
            failed.append((str(p), repr(e)))
        if verbose and (i + 1) % 500 == 0:
            print(f"  stats {i+1}/{len(paths)} failed={len(failed)}", flush=True)
    return out, keys, failed
