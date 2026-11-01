"""Generate synthetic data and train the cluster recogniser.

    python scripts/train.py --passes 30 --model mlp --out models/recognizer.pkl
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import synth  # noqa: E402
from src.recognize import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=30, help="synthetic samples per unit (before filtering)")
    ap.add_argument("--model", default="factor", choices=["factor", "mlp", "knn", "svm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/recognizer.pkl")
    ap.add_argument("--cache", default="models/synth_cache.npz")
    a = ap.parse_args()

    t = time.time()
    table, labels = synth.build_piece_table()
    print(f"piece table: {len(table)} units -> {len(labels)} piece labels")
    cache = Path(a.cache)
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        X, y = d["X"], d["y"]
        print(f"loaded cached synthetic data {X.shape}")
    else:
        X, y = synth.generate(a.passes, table, seed=a.seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y)
    # held-out synthetic validation (10%)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(y))
    nv = len(y) // 10
    va, tr = idx[:nv], idx[nv:]
    rec = train(table, X[tr], y[tr], kind=a.model, seed=a.seed)
    acc = float((rec.model.predict(X[va]) == y[va]).mean())
    print(f"synthetic held-out cluster accuracy: {acc:.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rec.save(a.out)
    print(f"saved {a.out}  (total {time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
