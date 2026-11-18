#!/usr/bin/env python3
"""
PathSnap: Standalone Top-1 Accuracy Evaluator
============================================
Evaluates and outputs strictly the Top-1 Accuracy:
- Pointwise Standalone Classifier Top-1 Accuracy
- HMM + Viterbi Sequence Top-1 Accuracy
- Net Accuracy Improvement (+Delta%)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from hmm_decoder import HMMViterbiMapMatcher

MODEL_FILES = {
    "xgboost": "model_xgboost.joblib",
    "svm": "model_svm.joblib",
    "random_forest": "model_random_forest.joblib",
    "logistic_regression": "model_logistic_regression.joblib",
    "knn": "model_knn.joblib"
}


def run_evaluation(model_name: str = "xgboost", trip_choice: str = "all"):
    model_key = model_name.lower().strip()
    model_filename = MODEL_FILES.get(model_key, model_key)
    model_path = Path(model_filename)

    if not model_path.exists():
        print(f"[!] Error: Model file {model_path} not found.")
        sys.exit(1)

    classifier = joblib.load(model_path)
    test_dir = Path("processed_enhanced/student_test")

    if not test_dir.exists():
        print(f"[!] Error: Preprocessed directory {test_dir} not found.")
        sys.exit(1)

    if trip_choice == "all":
        trip_ids = sorted([p.name for p in test_dir.glob("*") if p.is_dir() and (p / "candidate_features.csv").exists()])
    else:
        trip_ids = [trip_choice.strip()]

    print("=" * 80)
    print(f" Model Selected : {model_name.upper()} ({model_path.name})")
    print(f" Evaluation Set : {len(trip_ids)} official student evaluation trips")
    print("=" * 80)
    print(f" {'Trip ID':<12} | {'Points':<8} | {'Classifier Top-1':<18} | {'HMM Viterbi Top-1':<18} | {'Gain':<8}")
    print("-" * 80)

    matcher = HMMViterbiMapMatcher()
    total_pts = 0
    total_clf_hits = 0
    total_vit_hits = 0

    for tid in trip_ids:
        p = test_dir / tid
        cands = pd.read_csv(p / "candidate_features.csv")
        arcs = pd.read_csv(p / "arcs_features.csv")
        track = pd.read_csv(p / "track_features.csv")

        res = matcher.decode(cands, arcs, track, classifier, trip_id=tid)

        if res.classifier_top1_accuracy is not None:
            c_str = f"{res.classifier_top1_accuracy:.2f}%"
            v_str = f"{res.viterbi_top1_accuracy:.2f}%"
            g_str = f"+{res.accuracy_gain:.2f}%"
            print(f" {tid:<12} | {res.num_gps_points:<8} | {c_str:<18} | {v_str:<18} | {g_str:<8}")

            if res.ground_truth_arc_sequence:
                for c, v, g in zip(res.classifier_arc_sequence, res.viterbi_arc_sequence, res.ground_truth_arc_sequence):
                    if g != -1:
                        total_pts += 1
                        if c == g:
                            total_clf_hits += 1
                        if v == g:
                            total_vit_hits += 1

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PathSnap Top-1 Accuracy Evaluator")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "svm", "random_forest", "logistic_regression", "knn"],
        help="Model family to evaluate"
    )
    parser.add_argument(
        "--trip",
        type=str,
        default="all",
        help="Specific trip ID (e.g. 00000000) or 'all' for complete holdout set"
    )
    args = parser.parse_args()
    run_evaluation(args.model, args.trip)


if __name__ == "__main__":
    main()
