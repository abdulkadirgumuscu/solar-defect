"""Classify busbar slip from saved deep features with 1-NN and Naive Bayes.

Two evaluations:
  1. Hold-out (as in the paper): 375 training / 125 test images, stratified.
  2. Repeated stratified 5-fold cross-validation (10 repeats), which shows how
     much the scores move between splits. On a 125-image test set one image
     changes accuracy by 0.008, so single-split differences can be noise.

Usage:
    python -m src.classify --features features/features.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SINGLE = ["Relu4", "Relu25", "Relu34", "Conv10"]
PAIRS = [("Relu4", "Relu25"), ("Relu4", "Relu34"), ("Relu25", "Relu34"),
         ("Relu4", "Conv10"), ("Relu25", "Conv10"), ("Relu34", "Conv10")]


def load(path):
    d = np.load(path, allow_pickle=False)
    feats = {k: d[k] for k in SINGLE if k in d.files}
    return feats, d["labels"]


def feature_sets(feats):
    """Single layers, then pairs joined side by side (concatenation)."""
    sets = {name: feats[name] for name in SINGLE}
    for a, b in PAIRS:
        sets[f"{a} + {b}"] = np.hstack([feats[a], feats[b]])
    return sets


def classifiers(scale=False):
    clfs = {"k-NN (k=1)": KNeighborsClassifier(n_neighbors=1, metric="euclidean"),
            "Naive Bayes": GaussianNB()}
    if scale:
        clfs = {k: make_pipeline(StandardScaler(), v) for k, v in clfs.items()}
    return clfs


def scores(y_true, y_pred):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F-Score": f1_score(y_true, y_pred, zero_division=0)}


def holdout(sets, y, test_size=0.25, seed=42, scale=False):
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
    rows = []
    for clf_name, clf in classifiers(scale).items():
        for name, X in sets.items():
            clf.fit(X[idx_tr], y[idx_tr])
            pred = clf.predict(X[idx_te])
            rows.append({"Classifier": clf_name, "Features": name, "Dim.": X.shape[1], **scores(y[idx_te], pred)})
    return pd.DataFrame(rows), len(idx_tr), len(idx_te)


def cross_validate(sets, y, folds=5, repeats=10, seed=42, scale=False):
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y))
    rows = []
    for clf_name, clf in classifiers(scale).items():
        for name, X in sets.items():
            acc, f1 = [], []
            for tr, te in splits:
                pred = clf.fit(X[tr], y[tr]).predict(X[te])
                acc.append(accuracy_score(y[te], pred))
                f1.append(f1_score(y[te], pred, zero_division=0))
            rows.append({"Classifier": clf_name, "Features": name,
                         "Accuracy": f"{np.mean(acc):.4f} ± {np.std(acc):.4f}",
                         "F-Score": f"{np.mean(f1):.4f} ± {np.std(f1):.4f}",
                         "_mean_f1": np.mean(f1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/features.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", action="store_true", help="standardise features first (not used in the paper)")
    args = ap.parse_args()

    feats, y = load(args.features)
    sets = feature_sets(feats)
    print(f"{len(y)} images: {(y == 0).sum()} flawless, {(y == 1).sum()} busbar slip\n")

    hold, n_tr, n_te = holdout(sets, y, seed=args.seed, scale=args.scale)
    cv = cross_validate(sets, y, seed=args.seed, scale=args.scale)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Results\n\n{len(y)} images ({(y == 0).sum()} flawless, {(y == 1).sum()} busbar slip). "
          f"Positive class: busbar slip. Feature scaling: {'yes' if args.scale else 'no (as in the paper)'}.\n",
          f"## Hold-out: {n_tr} training / {n_te} test images (seed {args.seed})\n"]
    for clf_name, g in hold.groupby("Classifier", sort=False):
        md += [f"### {clf_name}\n", g.drop(columns="Classifier").round(4).to_markdown(index=False), ""]
    md += ["## Repeated stratified 5-fold cross-validation (10 repeats): mean ± std\n",
           cv.drop(columns="_mean_f1").to_markdown(index=False), ""]
    (out / "results.md").write_text("\n".join(md))
    hold.to_csv(out / "holdout.csv", index=False)
    cv.drop(columns="_mean_f1").to_csv(out / "cross_validation.csv", index=False)

    print(hold.round(4).to_string(index=False))
    best = cv.sort_values("_mean_f1", ascending=False).head(5).drop(columns="_mean_f1")
    print("\nBest 5 by cross-validated F-score:\n", best.to_string(index=False))
    print(f"\nSaved to {out}/results.md")


if __name__ == "__main__":
    main()
