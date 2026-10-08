from __future__ import annotations

import csv
from pathlib import Path

METHODS = ("contact_shadow", "contact_ink")


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def score_pen_down(pendown_csv: str | Path) -> dict:
    """Score shadow-gap and fresh-ink contact signals against the
    key-held ground truth column (M6 experiment harness)."""
    counts = {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for m in METHODS}
    frames = 0
    with Path(pendown_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            frames += 1
            truth = row.get("key_down") == "1"
            for m in METHODS:
                raw = row.get(m, "")
                if raw == "":
                    continue  # signal unavailable (e.g. no tip visible)
                pred = raw == "1"
                c = counts[m]
                if pred and truth:
                    c["tp"] += 1
                elif pred and not truth:
                    c["fp"] += 1
                elif not pred and truth:
                    c["fn"] += 1
                else:
                    c["tn"] += 1
    return {
        "frames": frames,
        "contact_shadow": _prf(**{k: counts["contact_shadow"][k] for k in ("tp", "fp", "fn")}),
        "contact_ink": _prf(**{k: counts["contact_ink"][k] for k in ("tp", "fp", "fn")}),
    }
