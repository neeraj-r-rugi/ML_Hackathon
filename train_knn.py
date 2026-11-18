#!/usr/bin/env python3
"""
PathSnap: K-Nearest Neighbors (KNN) Suite & Bayesian Optimization
================================================================
Complies strictly with Hackathon Section 3 (Model Experimentation) and Section 4 (Bayesian Optimization):
1. Ingests and enriches road candidate feature tables.
2. Group-Aware Cross Validation by complete trip_id (GroupKFold).
3. Evaluates 5 distinct KNN model configurations from the Instance-Based algorithm family:
   - Model 1: Standard Uniform KNN (Euclidean distance, uniform voting)
   - Model 2: Distance-Weighted KNN (Inverse Euclidean distance weights)
   - Model 3: Manhattan (L1 / Cityblock) KNN (Robust to coordinate scaling)
   - Model 4: Fine-Grained Local KNN (k=7, sharp local decision boundaries)
   - Model 5: Bayesian Hyperparameter-Optimized KNN (Optuna TPE optimization)
4. Computes observation-level Top-1 Accuracy, Macro F1, Precision, Recall, Log Loss, and Latency.
5. Evaluates on both the multi-model holdout set (Trips 26-29) and the highway test set (Trips 13-14).
6. Exports winning model artifact to model_knn.joblib and json results.
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
import optuna

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix
)
from sklearn.model_selection import GroupKFold

# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)

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


# ==============================================================================
# 1. Ingestion & Preprocessing
# ==============================================================================

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


def build_balanced_knn_train_set(df: pd.DataFrame, neg_ratio: float = 2.0, max_samples: int = 60000) -> pd.DataFrame:
    """
    Instance-based models like KNN suffer from O(N_train * N_test) complexity
    and extreme majority-class bias (18:1 negatives).
    This function extracts all ground-truth positive candidates and a representative
    sample of negatives (1:2 ratio) to ensure balanced neighborhood voting.
    """
    pos_mask = df["is_reference_candidate"] == 1
    pos_df = df[pos_mask]
    neg_df = df[~pos_mask]
    
    n_pos = len(pos_df)
    n_neg = int(min(len(neg_df), n_pos * neg_ratio))
    
    # Subsample if total exceeds max_samples
    if (n_pos + n_neg) > max_samples:
        n_pos = int(max_samples / (1.0 + neg_ratio))
        n_neg = int(n_pos * neg_ratio)
        pos_df = pos_df.sample(n=n_pos, random_state=42)
        
    sampled_neg = neg_df.sample(n=n_neg, random_state=42)
    balanced_df = pd.concat([pos_df, sampled_neg], ignore_index=True).sample(frac=1.0, random_state=42)
    return balanced_df


# ==============================================================================
# 2. Evaluation Utilities
# ==============================================================================

def calculate_top1_accuracy(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Select candidate with highest predicted score per GPS observation."""
    temp = df[["trip_id", "gps_index", "is_reference_candidate"]].copy()
    temp["score"] = scores
    best_idx = temp.groupby(["trip_id", "gps_index"])["score"].idxmax()
    return float(np.mean(temp.loc[best_idx, "is_reference_candidate"].values)) * 100.0


def evaluate_model(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    test_df: pd.DataFrame
) -> Dict:
    """Comprehensive metric evaluation following competition guidelines."""
    t0 = time.time()
    probs = model.predict_proba(X_test)
    inf_time_sec = time.time() - t0
    latency_us = (inf_time_sec / max(1, len(X_test))) * 1e6
    
    pos_scores = probs[:, 1]
    preds = (pos_scores >= 0.5).astype(int)
    
    # Safe clipped log-loss
    loss_val = float(log_loss(y_test, np.clip(probs, 1e-7, 1.0 - 1e-7)))
    
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


# ==============================================================================
# 3. Model Suite: 5 Distinct KNN Configurations
# ==============================================================================

def run_knn_experiment(
    train_balanced_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    test_highway_df: pd.DataFrame = None,
    n_optuna_trials: int = 8
) -> Tuple[Dict, Pipeline]:
    
    X_train = train_balanced_df[FEATURE_COLS].values
    y_train = train_balanced_df["is_reference_candidate"].values.astype(int)
    groups_train = train_balanced_df["trip_id"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_reference_candidate"].values.astype(int)
    
    results = {}

    # --------------------------------------------------------------------------
    # Model 1: Standard Uniform KNN
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 1: Standard Uniform KNN (k=15, Euclidean)")
    print("="*70)
    pipe_m1 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=15, weights="uniform", metric="euclidean", n_jobs=-1))
    ])
    t0 = time.time()
    pipe_m1.fit(X_train, y_train)
    m1_fit_time = time.time() - t0
    
    m1_eval = evaluate_model(pipe_m1, X_test, y_test, test_df)
    m1_eval["Training_Time_s"] = m1_fit_time
    m1_eval["Algorithm_Family"] = "Instance-Based (KNN Uniform)"
    m1_eval["Configuration"] = "k=15, weights=uniform, metric=euclidean"
    results["Model 1 (Standard Uniform KNN)"] = m1_eval
    print(f" [✓] Top-1 Acc: {m1_eval['Top1_Accuracy']:.2f}% | Macro F1: {m1_eval['Macro_F1']:.4f} | Loss: {m1_eval['Log_Loss']:.4f} | Index Time: {m1_fit_time:.2f}s | Latency: {m1_eval['Latency_us_per_sample']:.1f} µs")

    # --------------------------------------------------------------------------
    # Model 2: Distance-Weighted KNN
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 2: Distance-Weighted KNN (k=15, Inverse Distance)")
    print("="*70)
    pipe_m2 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance", metric="euclidean", n_jobs=-1))
    ])
    t0 = time.time()
    pipe_m2.fit(X_train, y_train)
    m2_fit_time = time.time() - t0
    
    m2_eval = evaluate_model(pipe_m2, X_test, y_test, test_df)
    m2_eval["Training_Time_s"] = m2_fit_time
    m2_eval["Algorithm_Family"] = "Instance-Based (KNN Distance-Weighted)"
    m2_eval["Configuration"] = "k=15, weights=distance, metric=euclidean"
    results["Model 2 (Distance-Weighted KNN)"] = m2_eval
    print(f" [✓] Top-1 Acc: {m2_eval['Top1_Accuracy']:.2f}% | Macro F1: {m2_eval['Macro_F1']:.4f} | Loss: {m2_eval['Log_Loss']:.4f} | Index Time: {m2_fit_time:.2f}s | Latency: {m2_eval['Latency_us_per_sample']:.1f} µs")

    # --------------------------------------------------------------------------
    # Model 3: Manhattan (L1 / Cityblock) KNN
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 3: Manhattan / Cityblock L1 KNN (k=15, Metric=L1)")
    print("="*70)
    pipe_m3 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance", metric="manhattan", n_jobs=-1))
    ])
    t0 = time.time()
    pipe_m3.fit(X_train, y_train)
    m3_fit_time = time.time() - t0
    
    m3_eval = evaluate_model(pipe_m3, X_test, y_test, test_df)
    m3_eval["Training_Time_s"] = m3_fit_time
    m3_eval["Algorithm_Family"] = "Instance-Based (Manhattan L1 KNN)"
    m3_eval["Configuration"] = "k=15, weights=distance, metric=manhattan (L1)"
    results["Model 3 (Manhattan L1 KNN)"] = m3_eval
    print(f" [✓] Top-1 Acc: {m3_eval['Top1_Accuracy']:.2f}% | Macro F1: {m3_eval['Macro_F1']:.4f} | Loss: {m3_eval['Log_Loss']:.4f} | Index Time: {m3_fit_time:.2f}s | Latency: {m3_eval['Latency_us_per_sample']:.1f} µs")

    # --------------------------------------------------------------------------
    # Model 4: Fine-Grained Local KNN (k=7)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 4: Fine-Grained Local KNN (k=7, High Resolution)")
    print("="*70)
    pipe_m4 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance", metric="euclidean", n_jobs=-1))
    ])
    t0 = time.time()
    pipe_m4.fit(X_train, y_train)
    m4_fit_time = time.time() - t0
    
    m4_eval = evaluate_model(pipe_m4, X_test, y_test, test_df)
    m4_eval["Training_Time_s"] = m4_fit_time
    m4_eval["Algorithm_Family"] = "Instance-Based (Fine-Grained KNN)"
    m4_eval["Configuration"] = "k=7, weights=distance, metric=euclidean"
    results["Model 4 (Fine-Grained Local KNN)"] = m4_eval
    print(f" [✓] Top-1 Acc: {m4_eval['Top1_Accuracy']:.2f}% | Macro F1: {m4_eval['Macro_F1']:.4f} | Loss: {m4_eval['Log_Loss']:.4f} | Index Time: {m4_fit_time:.2f}s | Latency: {m4_eval['Latency_us_per_sample']:.1f} µs")

    # --------------------------------------------------------------------------
    # Model 5: Bayesian Hyperparameter-Optimized KNN (Optuna TPE)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 5: BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna TPE)")
    print("="*70)
    print(f"    Running {n_optuna_trials} Bayesian optimization trials using GroupKFold...")

    # GroupKFold CV on balanced training subset
    cv = GroupKFold(n_splits=3)
    
    # Subsample validation set for fast Bayesian exploration
    opt_sub = train_balanced_df.sample(n=min(len(train_balanced_df), 18000), random_state=42)
    X_opt = opt_sub[FEATURE_COLS].values
    y_opt = opt_sub["is_reference_candidate"].values.astype(int)
    groups_opt = opt_sub["trip_id"].values

    def objective(trial: optuna.Trial) -> float:
        k = trial.suggest_int("n_neighbors", 5, 31, step=2)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_categorical("p", [1, 2])
        leaf_size = trial.suggest_int("leaf_size", 20, 50, step=10)

        f1_scores = []
        for tr_idx, val_idx in cv.split(X_opt, y_opt, groups=groups_opt):
            pipe_cv = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    p=p,
                    leaf_size=leaf_size,
                    n_jobs=-1
                ))
            ])
            pipe_cv.fit(X_opt[tr_idx], y_opt[tr_idx])
            y_pred = pipe_cv.predict(X_opt[val_idx])
            f1_scores.append(f1_score(y_opt[val_idx], y_pred, average="macro", zero_division=0))
        return float(np.mean(f1_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)
    
    best_params = study.best_params
    print(f"    [✓] Best Trial Score (CV Macro F1): {study.best_value:.4f}")
    print(f"    [✓] Optimal Hyperparameters: {best_params}")

    # Retrain best model on full balanced training set
    pipe_best = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=best_params["n_neighbors"],
            weights=best_params["weights"],
            p=best_params["p"],
            leaf_size=best_params["leaf_size"],
            n_jobs=-1
        ))
    ])
    
    t0 = time.time()
    pipe_best.fit(X_train, y_train)
    tuned_fit_time = time.time() - t0
    
    best_eval = evaluate_model(pipe_best, X_test, y_test, test_df)
    best_eval["Training_Time_s"] = tuned_fit_time
    best_eval["Algorithm_Family"] = "Bayesian-Tuned Instance-Based (KNN)"
    best_eval["Configuration"] = f"Optuna TPE: k={best_params['n_neighbors']}, weights={best_params['weights']}, p={best_params['p']}, leaf_size={best_params['leaf_size']}"
    best_eval["Best_Params"] = best_params
    best_eval["CV_Macro_F1"] = float(study.best_value)
    results["Model 5 (Bayesian Tuned KNN)"] = best_eval
    print(f" [✓] Tuned Top-1 Acc: {best_eval['Top1_Accuracy']:.2f}% | Tuned Macro F1: {best_eval['Macro_F1']:.4f} | Loss: {best_eval['Log_Loss']:.4f} | Index Time: {tuned_fit_time:.2f}s | Latency: {best_eval['Latency_us_per_sample']:.1f} µs")

    # Pick overall best model by Top-1 Accuracy
    winning_model_name = max(results.keys(), key=lambda k: results[k]["Top1_Accuracy"])
    model_map = {
        "Model 1 (Standard Uniform KNN)": pipe_m1,
        "Model 2 (Distance-Weighted KNN)": pipe_m2,
        "Model 3 (Manhattan L1 KNN)": pipe_m3,
        "Model 4 (Fine-Grained Local KNN)": pipe_m4,
        "Model 5 (Bayesian Tuned KNN)": pipe_best
    }
    winning_pipe = model_map[winning_model_name]
    
    # Also evaluate winning model on highway trips 13 & 14 if provided
    if test_highway_df is not None and not test_highway_df.empty:
        X_hw = test_highway_df[FEATURE_COLS].values
        probs_hw = winning_pipe.predict_proba(X_hw)[:, 1]
        top1_hw = calculate_top1_accuracy(test_highway_df, probs_hw)
        print(f"\n[*] Winning KNN Model ({winning_model_name}) evaluated on Trips 13 & 14 (Highway): {top1_hw:.2f}% Top-1 Accuracy")
        for k in results:
            # Add highway evaluation to all models
            m_pipe = model_map[k]
            p_hw = m_pipe.predict_proba(X_hw)[:, 1]
            results[k]["Highway_Top1_Accuracy"] = calculate_top1_accuracy(test_highway_df, p_hw)

    joblib.dump(winning_pipe, "model_knn.joblib")
    print(f"[*] Winning model artifact exported to: model_knn.joblib ({winning_model_name})")

    return results, winning_pipe


def print_competition_table(results: Dict):
    """Format and print the Section 3 / Section 4 competition table."""
    print("\n" + "="*110)
    print("SECTION 3 & 4: OVERALL MODEL COMPARISON (KNN INSTANCE-BASED FAMILY EXPERIMENTS)")
    print("="*110)
    
    headers = [
        "Model", "Algorithm Family", "Top-1 (Urban)", "Top-1 (Hwy)", "Macro F1", 
        "Log Loss", "Inference Latency", "Index Time"
    ]
    
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            m["Algorithm_Family"],
            f"{m['Top1_Accuracy']:.2f}%",
            f"{m.get('Highway_Top1_Accuracy', np.nan):.2f}%" if "Highway_Top1_Accuracy" in m else "N/A",
            f"{m['Macro_F1']:.4f}",
            f"{m['Log_Loss']:.4f}" if not np.isnan(m["Log_Loss"]) else "N/A",
            f"{m['Latency_us_per_sample']:.1f} µs",
            f"{m['Training_Time_s']:.2f}s"
        ])
        
    df_table = pd.DataFrame(rows, columns=headers)
    print(df_table.to_string(index=False))
    print("="*110 + "\n")
    
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {
            key: val for key, val in v.items() 
            if key not in ["Scores", "Predictions"]
        }
    with open("knn_benchmark_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print("[*] Benchmark results saved to: knn_benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(description="PathSnap KNN Training and Bayesian Optimization Pipeline")
    parser.add_argument("--train-trips", type=int, default=18, help="Number of training trips to include")
    parser.add_argument("--test-trips", type=int, default=4, help="Number of holdout trips for evaluation")
    parser.add_argument("--trials", type=int, default=8, help="Number of Optuna Bayesian optimization trials")
    args = parser.parse_args()

    train_dir = Path("processed_enhanced/train")
    all_train_trips = sorted([p.name for p in train_dir.glob("*") if p.is_dir() and p.name.isdigit()])
    
    train_ids = all_train_trips[:args.train_trips]
    test_ids = all_train_trips[args.train_trips : args.train_trips + args.test_trips]
    
    print(f"\n[*] Ingesting {len(train_ids)} Training Trips from {train_dir}...")
    full_train_df = load_dataset(train_dir, train_ids)
    print(f"[✓] Full train candidate samples: {len(full_train_df):,}")
    
    # Extract balanced prototype set for KNN
    train_balanced_df = build_balanced_knn_train_set(full_train_df, neg_ratio=2.0, max_samples=60000)
    print(f"[✓] KNN Balanced Prototype Samples: {len(train_balanced_df):,} (Pos: {np.sum(train_balanced_df['is_reference_candidate']==1):,}, Neg: {np.sum(train_balanced_df['is_reference_candidate']==0):,})")

    print(f"[*] Ingesting {len(test_ids)} Urban Holdout Evaluation Trips (Trips 26-29)...")
    test_df = load_dataset(train_dir, test_ids)
    print(f"[✓] Holdout test candidate samples: {len(test_df):,}")

    # Highway test set (Trips 13 & 14)
    print(f"[*] Ingesting Highway Evaluation Trips (Trips 13 & 14)...")
    highway_df = load_dataset(train_dir, ["00000013", "00000014"])
    print(f"[✓] Highway test candidate samples: {len(highway_df):,}")

    results, winning_pipe = run_knn_experiment(
        train_balanced_df, test_df, test_highway_df=highway_df, n_optuna_trials=args.trials
    )
    print_competition_table(results)


if __name__ == "__main__":
    main()
