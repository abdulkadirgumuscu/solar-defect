"""Tests for the classification stage, using synthetic features (no PyTorch, no images).

Run with `pytest` or `python tests/test_classify.py`.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify import PAIRS, SINGLE, cross_validate, feature_sets, holdout, load  # noqa: E402

DIMS = {"Relu4": 256, "Relu25": 1024, "Relu34": 1024, "Conv10": 1000}


def fake_features(n_ok=300, n_slip=200, signal=1.0, seed=0):
    """Same shapes and class balance as the paper; class 1 is shifted by `signal`."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_ok + [1] * n_slip)
    feats = {k: rng.normal(size=(len(y), d)) + signal * y[:, None] * (rng.random(d) < 0.05) for k, d in DIMS.items()}
    return feats, y


def test_feature_set_dimensions_match_the_paper():
    feats, _ = fake_features()
    sets = feature_sets(feats)
    assert len(sets) == len(SINGLE) + len(PAIRS) == 10
    expected = {"Relu4": 256, "Relu25": 1024, "Relu34": 1024, "Conv10": 1000,
                "Relu4 + Relu25": 1280, "Relu4 + Relu34": 1280, "Relu25 + Relu34": 2048,
                "Relu4 + Conv10": 1256, "Relu25 + Conv10": 2024, "Relu34 + Conv10": 2024}
    assert {k: v.shape[1] for k, v in sets.items()} == expected


def test_holdout_split_is_375_125():
    feats, y = fake_features()
    df, n_tr, n_te = holdout(feature_sets(feats), y)
    assert (n_tr, n_te) == (375, 125)
    assert len(df) == 20  # 10 feature sets x 2 classifiers


def test_learns_signal_and_not_noise():
    feats, y = fake_features(signal=3.0)
    df, _, _ = holdout(feature_sets(feats), y)
    assert df["Accuracy"].max() > 0.9
    feats, y = fake_features(signal=0.0)
    df, _, _ = holdout(feature_sets(feats), y)
    assert df["Accuracy"].max() < 0.75     # no signal -> near chance (0.6 majority class)


def test_cross_validation_and_npz_roundtrip():
    feats, y = fake_features(signal=3.0)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "f.npz"
        np.savez_compressed(path, labels=y, paths=np.array([f"img{i}.png" for i in range(len(y))]), **feats)
        feats2, y2 = load(path)
    assert (y2 == y).all() and set(feats2) == set(DIMS)
    cv = cross_validate(feature_sets(feats2), y2, repeats=2)
    assert len(cv) == 20 and cv["_mean_f1"].between(0, 1).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
