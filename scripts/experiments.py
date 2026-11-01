"""Ablations on the final pipeline. Writes results/experiments.tsv.

    python scripts/experiments.py      (~15 min on 2 CPUs; reuses models/synth_cache.npz)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import synth  # noqa: E402
from src.recognize import Recognizer, train  # noqa: E402


def real_cer(model_path, prior=True):
    env_args = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--model", str(model_path), "--out", "/tmp/exp"]
    if not prior:
        env_args.append("--no-prior")
    subprocess.run(env_args, capture_output=True, text=True, check=True)
    return json.load(open("/tmp/exp/eval.json"))["summary"]["CER"]


def run(name, X, y, table, kind="factor", prior=True, log=None):
    idx = np.random.default_rng(0).permutation(len(y))
    nv = len(y) // 10
    t = time.time()
    rec = train(table, X[idx[nv:]], y[idx[nv:]], kind=kind, verbose=False)
    dt = time.time() - t
    acc = float((rec.model.predict(X[idx[:nv]]) == y[idx[:nv]]).mean())
    p = ROOT / "models/_exp.pkl"
    rec.save(p)
    cer = real_cer(p, prior)
    line = f"{name}\t{len(y) - nv}\t{acc:.3f}\t{cer:.3f}\t{dt:.0f}"
    print(line, flush=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")


def main():
    log = ROOT / "results/experiments.tsv"
    log.write_text("experiment\ttrain_clusters\tsynthetic_acc\treal_CER\ttrain_s\n")
    table, _ = synth.build_piece_table()
    d = np.load(ROOT / "models/synth_cache.npz")
    X, y = d["X"], d["y"]
    run("final: factorised MLP, 100 renders/unit, layout prior", X, y, table, log=log)
    run("no digit layout prior", X, y, table, prior=False, log=log)
    run("flat MLP (one class per base+marks)", X, y, table, kind="mlp", log=log)
    run("kNN (k=3)", X, y, table, kind="knn", log=log)
    keep = np.random.default_rng(1).permutation(len(y))[: int(len(y) * 0.3)]
    run("30% of the synthetic data", X[keep], y[keep], table, log=log)
    Xc, yc = synth.generate(30, table, seed=5, verbose=False, do_degrade=False)
    run("clean renders, no degradation (30/unit)", Xc, yc, table, log=log)
    (ROOT / "models/_exp.pkl").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
