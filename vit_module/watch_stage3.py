"""Stage-3 训练监控脚本 — 查看训练进度和 loss。

用法:
    python vit_module/watch_stage3.py              # 查看当前状态一次
    python vit_module/watch_stage3.py --follow     # 持续监控 (Ctrl+C 退出)

说明:
    - 训练输出文件在 Claude Code 的临时目录 (后台任务 bnco4gboh.output)
    - 显示: 进度、真实 loss (DBG-LOSS)、trainer 日志、checkpoint 状态
"""
import os, sys, glob, time, re

OUTPUT_FILE = r"C:\Users\Supor2\AppData\Local\Temp\claude\E--Cross-domain-authentication-verification-Next-work-M2F2-Det-main-hyy\8cb51d32-84e6-4db6-9897-425c07d7fe57\tasks\bf0bv4t2x.output"
OUTPUT_DIR = "./checkpoints/llava-v1.5-7b-deepfake_stage-3-delta"

def print_status():
    if not os.path.exists(OUTPUT_FILE):
        print(f"[WARN] 找不到训练输出文件: {OUTPUT_FILE}")
        return
    with open(OUTPUT_FILE, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 1. 进度
    prog = None
    for line in lines:
        m = re.search(r"(\d+)/1734", line)
        if m:
            prog = m.group(1)
    print(f"进度: {prog}/1734" if prog else "进度: 解析中...")

    # 2. DBG-LOSS 最近值
    dbg_losses = [l.split("lm_loss.item()=")[1].split(" ")[0] for l in lines if "DBG-LOSS" in l]
    if dbg_losses:
        recent = dbg_losses[-10:]
        print(f"真实 loss (最近 {len(recent)} 步): {', '.join(recent)}")
        print(f"  min={min(float(x) for x in dbg_losses):.4f} max={max(float(x) for x in dbg_losses):.4f}")

    # 3. trainer 日志
    t_logs = [l[l.index("'loss"):l.index("}")+1] for l in lines if "'loss" in l and "learning_rate" in l]
    if t_logs:
        print(f"trainer 日志 (最近 {len(t_logs[-3:])} 条): {t_logs[-3:]}")

    # 4. checkpoint
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    if ckpts:
        print(f"checkpoints: {[os.path.basename(c) for c in ckpts]}")
    else:
        print("checkpoints: 无（未到 100 步）")

    # 5. 错误检查
    errs = [l.strip() for l in lines if "Error" in l or "Traceback" in l or "nan" in l.lower()]
    if errs:
        print(f"[!!] 检测到错误: {errs[-3:]}")
    else:
        print("状态: 正常（无错误）")

def main():
    follow = "--follow" in sys.argv
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 50)
        print("Stage-3 微调监控")
        print("=" * 50)
        print_status()
        if not follow:
            break
        print("\n(刷新中... Ctrl+C 退出)")
        time.sleep(30)

if __name__ == "__main__":
    main()
