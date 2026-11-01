"""Stage 2: glyph-cluster recogniser trained only on synthetic Noto renders.

Model: standardised pixel patch (32x40) -> PCA(120) -> classifier (MLP by default; kNN and
linear SVM kept for the experiments table). Saved as a single pickle with the piece table.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .clusters import band, clusters, gaps, normalize_patch
from .mapping import Mapper


def split_label(lab: str) -> tuple[str, str]:
    """'ka -e -m' -> ('ka', '-e -m');  'ka -o#0' -> ('ka', '-o#0');  'ga#1' -> ('ga#1', '')"""
    if " " in lab and not lab.startswith("-"):
        b, m = lab.split(" ", 1)
        return b, m
    return lab, ""


class FactorizedClassifier:
    """Two MLPs on shared features: base glyph and attached marks. Prediction = argmax over
    *valid* labels of log p(base) + log p(marks). ~10x more examples per class than a flat
    classifier over every base+mark combination."""

    def __init__(self, seed: int = 0, hidden: int = 512):
        self.feat = make_pipeline(StandardScaler(), PCA(120, random_state=seed))
        kw = dict(hidden_layer_sizes=(hidden,), alpha=1e-3, batch_size=256, max_iter=80,
                  early_stopping=True, n_iter_no_change=6, random_state=seed)
        self.base, self.mod = MLPClassifier(**kw), MLPClassifier(**kw)

    def fit(self, X, y):
        F = self.feat.fit_transform(X)
        parts = [split_label(l) for l in y]
        self.base.fit(F, [p[0] for p in parts])
        self.mod.fit(F, [p[1] for p in parts])
        self.labels = sorted(set(y))
        bi = {c: i for i, c in enumerate(self.base.classes_)}
        mi = {c: i for i, c in enumerate(self.mod.classes_)}
        self._bidx = np.array([bi[split_label(l)[0]] for l in self.labels])
        self._midx = np.array([mi[split_label(l)[1]] for l in self.labels])
        return self

    def scores(self, X):
        F = self.feat.transform(X)
        lb = np.log(self.base.predict_proba(F) + 1e-9)
        lm = np.log(self.mod.predict_proba(F) + 1e-9)
        return lb[:, self._bidx] + lm[:, self._midx]

    def predict(self, X):
        return np.array(self.labels)[self.scores(X).argmax(1)]


def make_model(kind: str = "factor", seed: int = 0):
    if kind == "factor":
        return FactorizedClassifier(seed)
    if kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
    elif kind == "svm":
        clf = LinearSVC(C=0.05, max_iter=3000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3, batch_size=256, max_iter=60,
                            early_stopping=True, n_iter_no_change=6, random_state=seed)
    return make_pipeline(StandardScaler(), PCA(120, random_state=seed), clf)


class Recognizer:
    def __init__(self, model, table):
        self.model, self.table = model, table
        self.mapper = Mapper(table)

    @classmethod
    def load(cls, path: str | Path) -> "Recognizer":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["model"], d["table"])

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"model": self.model, "table": self.table}, fh)

    DIGIT_PRIOR = 3.0   # log-score bonus/penalty for digits by layout (see RETRO.md)

    def labels(self, gray: np.ndarray, layout_prior: bool = True):
        boxes, ink, body = clusters(gray)
        if not boxes:
            return [], [], body, boxes
        y0, y1 = band(boxes, body, ink)
        X = np.stack([normalize_patch(ink, b, y0, y1).ravel() for b in boxes])
        gp = gaps(boxes)
        if not (layout_prior and hasattr(self.model, "scores")):
            return list(self.model.predict(X)), gp, body, boxes
        S = self.model.scores(X)
        labs = np.array(self.model.labels)
        is_digit = np.array([l.isdigit() for l in labs])
        sp = self.mapper and 0.55 * body
        for i in range(len(boxes)):
            left = i == 0 or gp[i - 1] > sp
            right = i == len(boxes) - 1 or gp[i] > sp
            # numerals stand alone (verse / item numbers); letters live inside words
            S[i, is_digit] += self.DIGIT_PRIOR if (left and right) else -self.DIGIT_PRIOR
        return list(labs[S.argmax(1)]), gp, body, boxes

    def recognize(self, gray: np.ndarray) -> str:
        labs, gp, body, _ = self.labels(gray)
        return self.mapper.to_unicode(labs, gp, body) if labs else ""


def train(table, X, y, kind="factor", seed=0, verbose=True):
    t = time.time()
    model = make_model(kind, seed)
    model.fit(X, y)
    if verbose:
        print(f"  trained {kind} on {len(y)} samples / {len(set(y))} classes in {time.time() - t:.0f}s")
    return Recognizer(model, table)
