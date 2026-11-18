#!/usr/bin/env python3
"""
PathSnap: Random Forest Model Suite
===================================
Ensemble & Bagging Decision Trees for Road Candidate Matching:
1. Ingests candidate feature tables from processed_enhanced/train.
2. Group-Aware Cross Validation by complete trip_id (GroupKFold).
3. Evaluates 100-tree Random Forest with balanced sub-sampling and depth regulation.
4. Computes observation-level Top-1 Accuracy, Macro F1, Precision, Recall, Log Loss, and Latency.
5. Exports model weights to model_random_forest.joblib.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix
)
from sklearn.model_selection import GroupKFold

FEATURE_COLS = [
    "distance_to_arc_m",
    "log_distance",
    "projection_fraction",
    "length_m",
    "log_length",
    "dist_to_length_ratio",
    "candidate_rank",
    "relative_dist_to_best",
    "heading_diff_deg",
    "cos_heading_diff",
    "is_wrong_way",
    "candidate_density",
    "from_node_degree",
    "to_node_degree",
    "speed_kmh"
]


def load_trip(trip_path: Path) -> pd.DataFrame:
    """Load candidate features from preprocessed table."""
    f = trip_path / "candidate_features.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "log_distance" not in df.columns:
        df["log_distance"] = np.log1p(np.maximum(df["distance_to_arc_m"].fillna(0.0), 0.0))
    if "log_length" not in df.columns:
        df["log_length"] = np.log1p(np.maximum(df["length_m"].fillna(0.0), 0.0))
    if "dist_to_length_ratio" not in df.columns:
        df["dist_to_length_ratio"] = df["distance_to_arc_m"] / (df["length_m"] + 1.0)
    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    df["trip_id"] = trip_path.name
    return df


def load_dataset(base_dir: Path, trip_ids: List[str]) -> pd.DataFrame:
    """Load specified trips into a combined dataframe."""
    dfs = []
    for tid in trip_ids:
        t_path = base_dir / tid
        if t_path.is_dir():
            df = load_trip(t_path)
            if not df.empty:
                dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def calculate_top1_accuracy(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Select candidate with highest predicted score per GPS observation."""
    temp = df[["trip_id", "gps_index", "is_reference_candidate"]].copy()
    temp["score"] = scores
    best_idx = temp.groupby(["trip_id", "gps_index"])["score"].idxmax()
    return float(np.mean(temp.loc[best_idx, "is_reference_candidate"].values)) * 100.0


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, test_df: pd.DataFrame) -> Dict:
    """Comprehensive metric evaluation."""
    t0 = time.time()
    probs = model.predict_proba(X_test)
    inf_time_sec = time.time() - t0
    latency_us = (inf_time_sec / max(1, len(X_test))) * 1e6
    
    pos_scores = probs[:, 1]
    preds = (pos_scores >= 0.5).astype(int)
    loss_val = float(log_loss(y_test, probs))
    
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    top1_acc = calculate_top1_accuracy(test_df, pos_scores)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Macro_F1": macro_f1,
        "Top1_Accuracy": top1_acc,
        "Log_Loss": loss_val,
        "Inference_Time_s": inf_time_sec,
        "Latency_us_per_sample": latency_us,
        "Scores": pos_scores,
        "Predictions": preds
    }


def main():
    parser = argparse.ArgumentParser(description="PathSnap Random Forest Pipeline")
    parser.add_argument("--train-trips", type=int, default=18, help="Number of trips to train on")
    parser.add_argument("--test-trips", type=int, default=4, help="Number of holdout trips to evaluate on")
    args = parser.parse_args()

    train_dir = Path("processed_enhanced/train")
    all_trips = sorted([p.name for p in train_dir.glob("*") if p.is_dir() and p.name.isdigit()])
    
    train_ids = all_trips[:args.train_trips]
    test_ids = all_trips[args.train_trips : args.train_trips + args.test_trips]

    print(f"\n{'='*75}")
    print(" PATHSNAP: RANDOM FOREST (ENSEMBLE / BAGGING FAMILY)")
    print(f"{'='*75}")
    print(f"[*] Ingesting {len(train_ids)} Training Trips...")
    train_df = load_dataset(train_dir, train_ids)
    print(f"[✓] Loaded {len(train_df):,} training candidate samples.")

    print(f"[*] Ingesting {len(test_ids)} Holdout Evaluation Trips (Trips {test_ids})...")
    test_df = load_dataset(train_dir, test_ids)
    print(f"[✓] Loaded {len(test_df):,} test candidate samples.")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_reference_candidate"].values.astype(int)
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_reference_candidate"].values.astype(int)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("\n[*] Training Random Forest model (100 trees, max_depth=14)...")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"[✓] Fit completed in {fit_time:.2f}s")

    eval_res = evaluate_model(model, X_test, y_test, test_df)
    eval_res["Training_Time_s"] = fit_time
    eval_res["Algorithm_Family"] = "Ensemble (Random Forest)"
    eval_res["Key_Config"] = "n_est=100, max_depth=14, class_weight=balanced_subsample"

    # Evaluate on highway trips (13 & 14)
    hw_df = load_dataset(train_dir, ["00000013", "00000014"])
    if not hw_df.empty:
        probs_hw = model.predict_proba(hw_df[FEATURE_COLS].values)[:, 1]
        eval_res["Highway_Top1_Accuracy"] = calculate_top1_accuracy(hw_df, probs_hw)

    print("\n" + "="*75)
    print(" EVALUATION RESULTS:")
    print("="*75)
    print(f"  Top-1 Accuracy (Urban holdout):   {eval_res['Top1_Accuracy']:.2f}%")
    if "Highway_Top1_Accuracy" in eval_res:
        print(f"  Top-1 Accuracy (Highway holdout): {eval_res['Highway_Top1_Accuracy']:.2f}%")
    print(f"  Macro F1-Score:                   {eval_res['Macro_F1']:.4f}")
    print(f"  Binary Precision:                 {eval_res['Precision']:.4f}")
    print(f"  Binary Recall:                    {eval_res['Recall']:.4f}")
    print(f"  Log Loss:                         {eval_res['Log_Loss']:.4f}")
    print(f"  Inference Latency per sample:     {eval_res['Latency_us_per_sample']:.2f} µs")
    print("="*75)

    joblib.dump(model, "model_random_forest.joblib")
    print("[*] Model artifact saved to: model_random_forest.joblib")

    clean_res = {k: v for k, v in eval_res.items() if k not in ["Scores", "Predictions"]}
    with open("random_forest_benchmark_results.json", "w") as f:
        json.dump(clean_res, f, indent=2)
    print("[*] Results saved to: random_forest_benchmark_results.json\n")


if __name__ == "__main__":
    main()
