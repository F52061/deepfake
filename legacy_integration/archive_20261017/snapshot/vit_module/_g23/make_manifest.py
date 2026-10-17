"""Build a diagnostics manifest CSV from vit_module/_g16/layer_feats.npz.

Three conversions are required and each is explicit below; see WORKLOG D.51.

  1. split:  the npz stores a target domain's split as the domain name itself
             (e.g. 'cd1'), which extract.py rejects. Target domains -> 'test'.
  2. y:      the npz uses y=1 for REAL (paths under original_sequences), the
             diagnostics tool requires y=1 for FAKE. Labels are flipped.
  3. ffiw:   excluded. It holds a single video, and analyze.py's donor_indices
             raises on any domain/split with fewer than 2 videos.

ffiw is dropped rather than remapped: a one-video domain cannot produce a
valid video-level bootstrap interval, which is the same conclusion already
recorded for FFIW in FINDINGS P8.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NPZ = PROJECT_ROOT / "vit_module" / "_g16" / "layer_feats.npz"

# Per (domain, split) row budget for the smoke manifest. Every group keeps at
# least two distinct videos so analyze.py's donor shuffle can run.
SMOKE_SIZES = {
    ("ffpp", "train"): 64,
    ("ffpp", "test"): 200,
    ("cd1", "test"): 32,
    ("cd2", "test"): 32,
    ("dfdcp", "test"): 32,
    ("wild", "test"): 32,
}


def load_rows():
    archive = np.load(NPZ, allow_pickle=True)
    return {
        "path": [str(p) for p in archive["paths"]],
        "y": archive["y"].astype(int),                  # 1 = real in the npz
        "domain": [str(d) for d in archive["domain"]],
        "split": [str(s) for s in archive["split"]],
        "video_id": [str(v) for v in archive["vids"]],
    }


def convert(rows):
    """Apply the three documented conversions. Returns a list of dicts."""
    out = []
    for i in range(len(rows["path"])):
        domain, split = rows["domain"][i], rows["split"][i]
        if domain == "ffiw":
            continue
        out.append({
            "path": rows["path"][i],
            "y": 1 - rows["y"][i],                      # flip: 1 = fake
            "domain": domain,
            "split": split if split in {"train", "val", "test"} else "test",
            "video_id": rows["video_id"][i],
        })
    return out


def stratified_sample(rows, sizes):
    """Take roughly `sizes[group]` rows per group, whole videos at a time.

    Videos are added as units so every group spans >=2 videos. For the source
    training group the videos are drawn alternately from the fake and real
    pools, because analyze.py requires both classes in the training rows.
    """
    groups = {}
    for row in rows:
        groups.setdefault((row["domain"], row["split"]), []).append(row)

    picked = []
    for group, members in sorted(groups.items()):
        budget = sizes.get(group)
        if budget is None:
            raise ValueError(f"Smoke manifest has no size for group {group}")

        by_video = {}
        for row in members:
            by_video.setdefault(row["video_id"], []).append(row)
        for video in by_video:
            by_video[video].sort(key=lambda r: r["path"])

        # Alternate between the fake and real video pools wherever both exist.
        # Sorting by name alone would hand a whole group to one class, because
        # videos of the same class share a naming pattern.
        fake = sorted(v for v, rs in by_video.items() if rs[0]["y"] == 1)
        real = sorted(v for v, rs in by_video.items() if rs[0]["y"] == 0)
        needs_both_classes = bool(fake) and bool(real)
        if needs_both_classes:
            order = []
            for index in range(max(len(fake), len(real))):
                if index < len(fake):
                    order.append(fake[index])
                if index < len(real):
                    order.append(real[index])
        else:
            order = sorted(by_video)

        chosen, videos_used = [], 0
        for video in order:
            if len(chosen) >= budget and videos_used >= 2:
                break
            chosen.extend(by_video[video])
            videos_used += 1

        if videos_used < 2:
            raise ValueError(f"Group {group} produced {videos_used} video(s); need >=2")
        if needs_both_classes and len({r["y"] for r in chosen}) < 2:
            raise ValueError(f"Group {group} has only one class")
        picked.extend(chosen)
    return picked


def write_csv(path, rows):
    columns = ["path", "y", "domain", "split", "video_id"]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Write a small stratified subset instead of everything")
    args = parser.parse_args()

    rows = convert(load_rows())
    if args.smoke:
        rows = stratified_sample(rows, SMOKE_SIZES)

    # Guard the invariants the diagnostics tool will re-check, so a bad
    # manifest fails here rather than an hour into extraction.
    paths = [r["path"] for r in rows]
    assert len(set(paths)) == len(paths), "duplicate paths"
    assert all(Path(p).is_absolute() and Path(p).is_file() for p in paths), "missing image"
    assert {r["y"] for r in rows} == {0, 1}, "both classes required"
    ownership = {}
    for row in rows:
        key = (row["domain"], row["video_id"])
        if key in ownership and ownership[key] != row["split"]:
            raise AssertionError(f"video crosses splits: {key}")
        ownership[key] = row["split"]
    for row in rows:
        if row["split"] == "train" and row["domain"] != "ffpp":
            raise AssertionError("train rows must be in the source domain only")

    write_csv(args.output, rows)

    groups = {}
    for row in rows:
        groups.setdefault((row["domain"], row["split"]), []).append(row)
    print(f"wrote {args.output}: {len(rows)} rows")
    for group in sorted(groups):
        members = groups[group]
        videos = {r["video_id"] for r in members}
        fakes = sum(r["y"] for r in members)
        print(f"  {group[0]:8s} {group[1]:6s} n={len(members):5d} "
              f"videos={len(videos):4d} fake={fakes:5d} real={len(members) - fakes:5d}")


if __name__ == "__main__":
    main()
