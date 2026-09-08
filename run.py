"""Main pipeline: decode -> LLM stopping policy -> evaluate.

    python run.py --subjects 1 2 3 --looks 4 --window 0.3

For each subject: build honest per-look FBECCA evidence, run the LLM
stopping policy on every trial, then report accuracy, average looks used,
and information transfer rate (ITR).
"""
import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

from decode import build_multilook_evidence
from keyboard_map import CLASS_TO_CHAR
from llm_policy import llm_stopping_policy

GAZE_SHIFT_S = 0.5  # standard inter-selection overhead used in BCI ITR reporting


def wolpaw_itr(acc, n_classes, seconds_per_selection):
    if acc <= 0:
        bits = 0.0
    elif acc >= 1:
        bits = math.log2(n_classes)
    else:
        bits = (math.log2(n_classes) + acc * math.log2(acc)
                + (1 - acc) * math.log2((1 - acc) / (n_classes - 1)))
    return bits * 60 / (seconds_per_selection + GAZE_SHIFT_S)


def run_subject(subject: int, n_looks: int, window_s: float, max_workers: int = 16):
    looks, n_classes, n_blocks = build_multilook_evidence(subject, n_looks, window_s)

    def one_trial(args):
        true_class, block = args
        trial_scores = [looks[k][true_class, block] for k in range(n_looks)]
        pred, n_used = llm_stopping_policy(trial_scores, CLASS_TO_CHAR, requests.Session())
        return true_class, pred, n_used

    tasks = [(c, b) for c in range(n_classes) for b in range(n_blocks)]
    correct, total_looks = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for true_class, pred, n_used in ex.map(one_trial, tasks):
            correct += int(pred == true_class)
            total_looks += n_used

    acc = correct / len(tasks)
    avg_looks = total_looks / len(tasks)
    itr = wolpaw_itr(acc, n_classes, avg_looks * window_s)
    return {"subject": subject, "acc": acc, "avg_looks": avg_looks, "itr": itr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--looks", type=int, default=4, help="max looks available per trial")
    ap.add_argument("--window", type=float, default=0.3, help="seconds per look")
    args = ap.parse_args()

    results = [run_subject(s, args.looks, args.window) for s in args.subjects]
    for r in results:
        print(f"subject {r['subject']:>3}: acc={r['acc']*100:5.1f}%  "
              f"avg_looks={r['avg_looks']:.2f}/{args.looks}  ITR={r['itr']:.1f} bits/min")

    accs = np.array([r["acc"] for r in results])
    looks = np.array([r["avg_looks"] for r in results])
    itrs = np.array([r["itr"] for r in results])
    print(f"\nmean over {len(results)} subject(s): "
          f"acc={accs.mean()*100:.1f}±{accs.std()*100:.1f}%  "
          f"avg_looks={looks.mean():.2f}  ITR={itrs.mean():.1f}±{itrs.std():.1f} bits/min")


if __name__ == "__main__":
    main()
