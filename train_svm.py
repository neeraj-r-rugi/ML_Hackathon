#!/usr/bin/env python3
"""
PathSnap: Support Vector Machine (SVM) Suite & Bayesian Optimization
===================================================================
Complies with Hackathon Section 3 (Model Experimentation) and Section 4 (Bayesian Optimization):
1. Ingests and enriches road candidate feature tables.
2. Group-Aware Cross Validation by complete trip_id (GroupKFold).
3. Evaluates 5 distinct SVM model configurations from the kernel-based algorithm family:
   - Model 1: Linear SVM (LinearSVC with balanced class weights)
   - Model 2: Probabilistic SGD-SVM (Modified Huber Loss for calibrated probabilities)
   - Model 3: Non-Linear Kernel Approximation (Nystroem RBF + Linear SVM)
   - Model 4: Exact RBF Kernel SVM (SVC with RBF kernel on balanced candidate sample)
   - Model 5: Bayesian Hyperparameter-Optimized SVM (Optuna TPE optimization)
4. Computes observation-level Top-1 Accuracy, Macro F1, Precision, Recall, Log Loss, and Latency.
5. Exports winning model artifact and consolidated comparison table.
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
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
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
# 1. Feature Engineering & Ingestion
# ==============================================================================

def load_and_enrich_trip(trip_path: Path) -> pd.DataFrame:
    """Load enhanced candidate features directly from preprocessed table."""
    cand_file = trip_path / "candidate_features.csv"
    if not cand_file.exists():
        return pd.DataFrame()
        
    df = pd.read_csv(cand_file)
    df["log_distance"] = np.log1p(np.maximum(df["distance_to_arc_m"].fillna(0.0), 0.0))
    df["log_length"] = np.log1p(np.maximum(df["length_m"].fillna(0.0), 0.0))
    df["dist_to_length_ratio"] = df["dist_to_length_ratio"].fillna(0.0)
    
    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
            
    df["trip_id"] = trip_path.name
    return df


def load_dataset(base_dir: Path, max_trips: int = 15) -> pd.DataFrame:
    """Load a representative set of trips for training."""
    trip_dirs = sorted([p for p in base_dir.glob("*") if p.is_dir()])[:max_trips]
    dfs = []
    print(f"[*] Ingesting and engineering features from {len(trip_dirs)} trips in {base_dir.name}...")
    for p in trip_dirs:
        df = load_and_enrich_trip(p)
        if not df.empty:
            dfs.append(df)
            
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[✓] Total candidate observations loaded: {len(full_df):,}")
    print(f"    Positive route candidate rate: {full_df['is_reference_candidate'].mean()*100:.2f}%")
    return full_df


# ==============================================================================
# 2. Evaluation Utilities
# ==============================================================================

def calculate_top1_accuracy(df: pd.DataFrame, scores: np.ndarray) -> float:
    """
    Observation-level Top-1 Accuracy:
    Select candidate with maximum continuous decision score per GPS point
    and check if it matches the ground-truth route road segment.
    """
    temp_df = df[["trip_id", "gps_index", "is_reference_candidate"]].copy()
    temp_df["score"] = scores
    
    best_idx = temp_df.groupby(["trip_id", "gps_index"])["score"].idxmax()
    top1_hits = temp_df.loc[best_idx, "is_reference_candidate"].values
    return float(np.mean(top1_hits)) * 100.0


def evaluate_model(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    test_df: pd.DataFrame
) -> Dict:
    """Comprehensive metric evaluation following competition guidelines."""
    t0 = time.time()
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        pos_scores = probs[:, 1]
        loss_val = float(log_loss(y_test, probs))
    elif hasattr(model, "decision_function"):
        pos_scores = model.decision_function(X_test)
        # Platt/Sigmoid approximation for log loss reporting
        calib_probs = 1.0 / (1.0 + np.exp(-np.clip(pos_scores, -20, 20)))
        loss_val = float(log_loss(y_test, np.column_stack([1 - calib_probs, calib_probs])))
    else:
        pos_scores = model.predict(X_test).astype(float)
        loss_val = np.nan
        
    preds = model.predict(X_test)
    inf_time_sec = time.time() - t0
    latency_us = (inf_time_sec / max(1, len(X_test))) * 1e6
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    macro_f1 = f1_score(y_test, preds, average="macro")
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
# 3. Model Suite: 5 Distinct SVM Architectures
# ==============================================================================

def run_svm_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, n_optuna_trials: int = 12) -> Dict:
    """Train and evaluate 5 SVM models across linear, kernel approximation, and exact RBF."""
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_reference_candidate"].values.astype(int)
    groups_train = train_df["trip_id"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_reference_candidate"].values.astype(int)
    
    results = {}

    # --------------------------------------------------------------------------
    # Model 1: Linear SVM (LinearSVC with Balanced Class Weights)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 1: Linear SVM (LinearSVC)")
    print("="*70)
    # Subsample if training set is huge to guarantee fast linear convergence
    train_size = min(len(X_train), 150000)
    sub_idx = np.random.choice(len(X_train), train_size, replace=False) if len(X_train) > train_size else np.arange(len(X_train))
    
    m1_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=0.5, class_weight="balanced", dual="auto", max_iter=1500, random_state=42))
    ])
    t0 = time.time()
    m1_pipe.fit(X_train[sub_idx], y_train[sub_idx])
    m1_train_time = time.time() - t0
    
    m1_eval = evaluate_model(m1_pipe, X_test, y_test, test_df)
    m1_eval["Training_Time_s"] = m1_train_time
    m1_eval["Algorithm_Family"] = "Linear SVM"
    m1_eval["Configuration"] = "C=0.5, class_weight=balanced, dual=auto"
    results["Model 1 (LinearSVC)"] = m1_eval
    print(f" [✓] Top-1 Acc: {m1_eval['Top1_Accuracy']:.2f}% | Macro F1: {m1_eval['Macro_F1']:.4f} | Train Time: {m1_train_time:.2f}s")

    # --------------------------------------------------------------------------
    # Model 2: Probabilistic SGD-SVM (Modified Huber Loss)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 2: Probabilistic SGD-SVM (Modified Huber Loss)")
    print("="*70)
    m2_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("sgd_svm", SGDClassifier(
            loss="modified_huber",  # Calibrated probability output
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=1500,
            random_state=42
        ))
    ])
    t0 = time.time()
    m2_pipe.fit(X_train, y_train)
    m2_train_time = time.time() - t0
    
    m2_eval = evaluate_model(m2_pipe, X_test, y_test, test_df)
    m2_eval["Training_Time_s"] = m2_train_time
    m2_eval["Algorithm_Family"] = "SGD-SVM (Probabilistic)"
    m2_eval["Configuration"] = "loss=modified_huber, alpha=1e-4, class_weight=balanced"
    results["Model 2 (SGD Modified Huber)"] = m2_eval
    print(f" [✓] Top-1 Acc: {m2_eval['Top1_Accuracy']:.2f}% | Macro F1: {m2_eval['Macro_F1']:.4f} | Log Loss: {m2_eval['Log_Loss']:.4f}")

    # --------------------------------------------------------------------------
    # Model 3: Non-Linear Kernel Approximation (Nystroem RBF + Linear SVM)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 3: Non-Linear RBF Kernel Approximation (Nystroem + Linear SVM)")
    print("="*70)
    m3_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nystroem", Nystroem(kernel="rbf", gamma=0.1, n_components=100, random_state=42)),
        ("clf", SGDClassifier(loss="modified_huber", penalty="l2", alpha=1e-4, class_weight="balanced", random_state=42))
    ])
    t0 = time.time()
    m3_pipe.fit(X_train, y_train)
    m3_train_time = time.time() - t0
    
    m3_eval = evaluate_model(m3_pipe, X_test, y_test, test_df)
    m3_eval["Training_Time_s"] = m3_train_time
    m3_eval["Algorithm_Family"] = "RBF Kernel Approximation (Nystroem)"
    m3_eval["Configuration"] = "kernel=rbf, gamma=0.1, n_components=100, SGD solver"
    results["Model 3 (Nystroem RBF-SVM)"] = m3_eval
    print(f" [✓] Top-1 Acc: {m3_eval['Top1_Accuracy']:.2f}% | Macro F1: {m3_eval['Macro_F1']:.4f} | Train Time: {m3_train_time:.2f}s")

    # --------------------------------------------------------------------------
    # Model 4: Exact Non-Linear RBF Kernel SVM (SVC with RBF)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 4: Exact Non-Linear RBF Kernel SVM (SVC with RBF)")
    print("="*70)
    # To prevent O(N^3) memory/compute stall, we use a balanced candidate subset
    pos_mask = (y_train == 1)
    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(~pos_mask)[0]
    np.random.seed(42)
    # Cap to 12,000 for exact kernel dual fitting to avoid O(N^2) LibSVM compute explosion
    n_pos = min(len(pos_indices), 4000)
    n_neg = min(len(neg_indices), 8000)
    sub_pos = np.random.choice(pos_indices, n_pos, replace=False)
    sub_neg = np.random.choice(neg_indices, n_neg, replace=False)
    sub_indices = np.concatenate([sub_pos, sub_neg])
    np.random.shuffle(sub_indices)
    
    print(f"    Fitting exact dual support vectors on {len(sub_indices):,} balanced candidate pairs...")
    m4_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rbf_svc", SVC(C=1.5, kernel="rbf", gamma="scale", probability=True, random_state=42))
    ])
    t0 = time.time()
    m4_pipe.fit(X_train[sub_indices], y_train[sub_indices])
    m4_train_time = time.time() - t0
    
    m4_eval = evaluate_model(m4_pipe, X_test, y_test, test_df)
    m4_eval["Training_Time_s"] = m4_train_time
    m4_eval["Algorithm_Family"] = "Exact RBF Kernel SVM"
    m4_eval["Configuration"] = "C=1.5, kernel=rbf, gamma=scale, Platt probability calibration"
    results["Model 4 (Exact RBF-SVC)"] = m4_eval
    print(f" [✓] Top-1 Acc: {m4_eval['Top1_Accuracy']:.2f}% | Macro F1: {m4_eval['Macro_F1']:.4f} | Train Time: {m4_train_time:.2f}s")

    # --------------------------------------------------------------------------
    # Model 5: Bayesian Hyperparameter-Optimized SVM (Optuna TPE)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 5: BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna TPE)")
    print("="*70)
    gkf = GroupKFold(n_splits=3)
    
    # Subsample for fast cross-validation iterations
    opt_size = min(len(X_train), 60000)
    opt_idx = np.random.choice(len(X_train), opt_size, replace=False) if len(X_train) > opt_size else np.arange(len(X_train))
    X_opt = X_train[opt_idx]
    y_opt = y_train[opt_idx]
    groups_opt = groups_train[opt_idx]
    
    def objective(trial):
        gamma_val = trial.suggest_float("gamma", 1e-3, 0.5, log=True)
        n_comp = trial.suggest_int("n_components", 50, 150, step=25)
        alpha_val = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        
        cv_f1s = []
        for tr_idx, val_idx in gkf.split(X_opt, y_opt, groups=groups_opt):
            cv_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("nystroem", Nystroem(kernel="rbf", gamma=gamma_val, n_components=n_comp, random_state=42)),
                ("clf", SGDClassifier(
                    loss="modified_huber",
                    penalty="l2",
                    alpha=alpha_val,
                    class_weight="balanced",
                    random_state=42
                ))
            ])
            cv_pipe.fit(X_opt[tr_idx], y_opt[tr_idx])
            val_preds = cv_pipe.predict(X_opt[val_idx])
            cv_f1s.append(f1_score(y_opt[val_idx], val_preds, average="macro"))
            
        return float(np.mean(cv_f1s))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    print(f"    Running {n_optuna_trials} Bayesian optimization trials using GroupKFold...")
    study.optimize(objective, n_trials=n_optuna_trials)
    best_params = study.best_params
    print(f"    [✓] Best Trial Score (CV Macro F1): {study.best_value:.4f}")
    print(f"    [✓] Optimal Hyperparameters: {best_params}")

    # Refit tuned model
    m5_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nystroem", Nystroem(kernel="rbf", gamma=best_params["gamma"], n_components=best_params["n_components"], random_state=42)),
        ("clf", SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=best_params["alpha"],
            class_weight="balanced",
            random_state=42
        ))
    ])
    t0 = time.time()
    m5_pipe.fit(X_train, y_train)
    m5_train_time = time.time() - t0
    
    m5_eval = evaluate_model(m5_pipe, X_test, y_test, test_df)
    m5_eval["Training_Time_s"] = m5_train_time
    m5_eval["Algorithm_Family"] = "Bayesian-Tuned RBF-SVM"
    m5_eval["Configuration"] = f"Optuna TPE: gamma={best_params['gamma']:.4f}, n_comp={best_params['n_components']}, alpha={best_params['alpha']:.5f}"
    m5_eval["Best_Params"] = best_params
    m5_eval["CV_Macro_F1"] = float(study.best_value)
    results["Model 5 (Bayesian Tuned SVM)"] = m5_eval
    print(f" [✓] Tuned Top-1 Acc: {m5_eval['Top1_Accuracy']:.2f}% | Tuned Macro F1: {m5_eval['Macro_F1']:.4f}")

    # Export winning model artifact
    joblib.dump(m5_pipe, "svm_best_model.joblib")
    print(f"\n[*] Exported winning model artifact to: svm_best_model.joblib")

    return results


# ==============================================================================
# 4. Report Formatting & Section 3.1 Table
# ==============================================================================

def print_competition_table(results: Dict):
    """Format and print consolidated comparison table matching Section 3.1."""
    print("\n" + "="*95)
    print("SECTION 3.1: OVERALL MODEL COMPARISON (SVM FAMILY EXPERIMENTS)")
    print("="*95)
    
    headers = [
        "Model", "Algorithm Family", "Top-1 Acc (%)", "Macro F1", 
        "Log Loss", "Inference Latency", "Train Time (s)"
    ]
    
    rows = []
    for name, m in results.items():
        rows.append([
            name,
            m["Algorithm_Family"],
            f"{m['Top1_Accuracy']:.2f}%",
            f"{m['Macro_F1']:.4f}",
            f"{m['Log_Loss']:.4f}" if not np.isnan(m["Log_Loss"]) else "N/A",
            f"{m['Latency_us_per_sample']:.1f} µs",
            f"{m['Training_Time_s']:.2f}s"
        ])
        
    df_table = pd.DataFrame(rows, columns=headers)
    print(df_table.to_string(index=False))
    print("="*95 + "\n")
    
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {
            key: val for key, val in v.items() 
            if key not in ["Scores", "Predictions"]
        }
    with open("svm_benchmark_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print("[*] Benchmark results saved to: svm_benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(description="PathSnap SVM Training and Bayesian Optimization Pipeline")
    parser.add_argument("--train-trips", type=int, default=12, help="Number of training trips to include")
    parser.add_argument("--test-trips", type=int, default=3, help="Number of holdout trips for evaluation")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna Bayesian optimization trials")
    args = parser.parse_args()
    
    train_dir = Path("processed_enhanced/train")
    all_train_trips = sorted([p for p in train_dir.glob("*") if p.is_dir()])
    
    train_subset = all_train_trips[:args.train_trips]
    test_subset = all_train_trips[args.train_trips : args.train_trips + args.test_trips]
    
    # Load train data
    train_dfs = [load_and_enrich_trip(p) for p in train_subset]
    train_df = pd.concat([d for d in train_dfs if not d.empty], ignore_index=True)
    print(f"[✓] Train candidate samples: {len(train_df):,} from {len(train_subset)} trips")
    
    # Load test data
    test_dfs = [load_and_enrich_trip(p) for p in test_subset]
    test_df = pd.concat([d for d in test_dfs if not d.empty], ignore_index=True)
    print(f"[✓] Holdout test candidate samples: {len(test_df):,} from {len(test_subset)} trips")
    
    results = run_svm_experiment(train_df, test_df, n_optuna_trials=args.trials)
    print_competition_table(results)


if __name__ == "__main__":
    main()
