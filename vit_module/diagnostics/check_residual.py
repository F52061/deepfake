"""Synthetic checks, not empirical residual-fusion evidence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from residual_clip import residualize


def main():
    rng = np.random.default_rng(7)
    count = 120
    visual = rng.normal(size=(count, 8))
    clip = visual @ rng.normal(size=(8, 10)) + rng.normal(size=(count, 10)) * 0.3
    labels = (visual[:, 0] + clip[:, 0] * 0.1 > 0).astype(int)
    train = np.arange(count) < 60
    groups = np.array([f"video{index // 5}" for index in range(count)])
    first = residualize(visual, clip, train, groups, [1, 10])
    changed = clip.copy()
    changed[~train] += 100
    second = residualize(visual, changed, train, groups, [1, 10])
    np.testing.assert_array_equal(first[3].coef_, second[3].coef_)
    assert first[4:] == second[4:]
    np.testing.assert_allclose(first[0] + first[1], clip)
    with tempfile.TemporaryDirectory(prefix="deepfake-residual-check-") as temporary:
        directory = Path(temporary)
        features = directory / "features.npz"
        np.savez_compressed(features, V=visual, C=clip, y=labels, row_id=np.arange(count),
                            path=np.array([f"synthetic/{index}" for index in range(count)]),
                            domain=np.array(["ffpp"] * 60 + ["target"] * 60),
                            split=np.where(train, "train", "test"), video_id=groups)
        output = directory / "analysis"
        subprocess.run([sys.executable, str(Path(__file__).with_name("residual_clip.py")),
                        "--input", str(features), "--output", str(output), "--label-one", "fake",
                        "--alphas", "1", "10", "--bootstrap", "20", "--repeats", "2"], check=True)
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert config["equivalent_score_error"] < 1e-7
        with np.load(output / "scores.npz") as scores:
            np.testing.assert_allclose(scores["V_C"], scores["V_Cres_equivalent"], atol=1e-7)
        with np.load(output / "controls.npz") as controls:
            assert np.all(groups[controls["donors_0"]] != groups)
            assert np.all(train[controls["donors_0"]] == train)
        assert (output / "COMPLETE").exists()
    print("PASS: source-only ridge CV, reconstruction, equivalent scores, donor isolation, saved artifacts")


if __name__ == "__main__":
    main()
