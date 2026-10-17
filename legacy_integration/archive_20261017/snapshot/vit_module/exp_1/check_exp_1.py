"""Synthetic checks only; never evidence of empirical complementarity."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp_1_analysis", ROOT / "analyze.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StatisticsTests(unittest.TestCase):
    def fixture(self):
        size = 12
        labels = np.array([0, 1] * 6)
        vit = np.array([-2, 2] * 2 + [-2, -2, -2, -2, 2, 2, 2, 2], dtype=float)
        clip = np.array([-3, 3] * 2 + [-3, -3, 3, 3, -3, -3, 3, 3], dtype=float)
        return {"row_id": np.arange(size), "path": np.array([f"image_{index}" for index in range(size)]),
                "domain": np.array(["ffpp"] * 4 + ["cd2"] * 8),
                "split": np.array(["val"] * 4 + ["test"] * 8),
                "video_id": np.array([f"video_{index // 2}" for index in range(size)]),
                "group_id": np.array([f"group_{index // 2}" for index in range(size)]),
                "y": labels, "vit_score": vit, "clip_score": clip}

    def provenance(self):
        common = {"higher_score": "fake", "checkpoint_sha256": "0" * 64,
                  "evaluation_labels_used_for_training": False,
                  "validation_groups_excluded_from_training": True,
                  "preprocessing": "synthetic", "training_description": "synthetic",
                  "score_definition": "synthetic logit", "selection_history": "synthetic only"}
        return {"source": "ffpp", "label_definition": "1=fake", "video_identity_audited": True,
                "vit": {**common, "kind": "native"}, "clip": {**common, "kind": "clip_only_probe"}}

    def test_auc_ties(self):
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(MODULE.auc(labels, np.array([0, 0, 1, 1])), 1)
        self.assertEqual(MODULE.auc(labels, np.ones(4)), .5)
        self.assertEqual(MODULE.auc(labels, np.array([1, 1, 0, 0])), 0)
        self.assertIsNone(MODULE.auc(np.ones(4), np.ones(4)))
        rng = np.random.default_rng(33)
        for _ in range(20):
            scores = rng.integers(-2, 3, 4)
            positive = scores[labels == 1][:, None]
            negative = scores[labels == 0][None, :]
            expected = np.mean((positive > negative) + .5 * (positive == negative))
            self.assertEqual(MODULE.auc(labels, scores), expected)

    def test_four_groups_and_target_isolation(self):
        data = self.fixture()
        calibration = data["split"] == "val"
        args = argparse.Namespace(primary_domains=["cd2"], bootstrap=30, seed=13)
        summary, codes = MODULE.analyze(data, calibration, args)
        self.assertEqual(summary["thresholds"]["vit"]["value"], 0)
        self.assertEqual(summary["thresholds"]["clip"]["value"], 0)
        entry = summary["per_domain"]["cd2"]
        self.assertEqual([entry["groups"][name]["n"] for name in MODULE.GROUPS], [2, 2, 2, 2])
        self.assertEqual(set(codes[~calibration]), set(MODULE.GROUPS))
        self.assertEqual(entry["metrics"]["oracle_correctness_accuracy"], .75)
        changed = {name: value.copy() for name, value in data.items()}
        changed["y"][~calibration] = 1 - changed["y"][~calibration]
        changed["clip_score"][~calibration] *= -100
        changed["vit_score"][~calibration] += 100
        other, _ = MODULE.analyze(changed, calibration, args)
        self.assertEqual(summary["thresholds"], other["thresholds"])
        self.assertEqual(entry["bootstrap"]["delta_auc_clip_minus_vit"]["ci95"], [0, 0])

    def test_cli_and_rejections(self):
        with tempfile.TemporaryDirectory(prefix="selfcheck_", dir=ROOT) as temporary:
            directory = Path(temporary)
            data = self.fixture()
            input_path = directory / "input.npz"
            provenance_path = directory / "provenance.json"
            np.savez_compressed(input_path, **data)
            provenance_path.write_text(json.dumps(self.provenance()), encoding="utf-8")
            MODULE.load(input_path, provenance_path, "ffpp")
            rejected = self.provenance()
            rejected["vit"]["kind"] = "fused_detector"
            provenance_path.write_text(json.dumps(rejected), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent vit"):
                MODULE.load(input_path, provenance_path, "ffpp")
            provenance_path.write_text(json.dumps(self.provenance()), encoding="utf-8")
            output = directory / "run"
            command = [sys.executable, str(ROOT / "analyze.py"), "--input", str(input_path),
                       "--provenance", str(provenance_path), "--output", str(output),
                       "--primary-domains", "cd2", "--bootstrap", "30"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "统计结果.md").is_file())
            hashes = json.loads((output / "artifact_hashes.json").read_text())
            for name, expected in hashes.items():
                self.assertEqual(MODULE.digest(output / name), expected)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(set(np.load(output / "samples.npz")["error_group"][4:]), set(MODULE.GROUPS))
            data["domain"] = np.array(["ffpp"] * 12)
            data["video_id"][4] = data["video_id"][0]
            np.savez_compressed(input_path, **data)
            with self.assertRaisesRegex(ValueError, "crosses val/test"):
                MODULE.load(input_path, provenance_path, "ffpp")
            outside = command.copy()
            outside[outside.index("--output") + 1] = str(ROOT.parent / "forbidden_exp1_output")
            self.assertNotEqual(subprocess.run(outside, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
