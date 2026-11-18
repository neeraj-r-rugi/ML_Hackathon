#!/usr/bin/env python3
"""
PathSnap: XGBoost Gradient Boosting Suite & Bayesian Optimization
================================================================
Complies strictly with Hackathon Section 3 (Model Experimentation) and Section 4 (Bayesian Optimization):
1. Ingests and enriches road candidate feature tables.
2. Group-Aware Cross Validation by complete trip_id (GroupKFold).
3. Evaluates 5 distinct XGBoost model configurations from the boosting algorithm family:
   - Model 1: Baseline XGBoost (XGBClassifier with scale_pos_weight)
   - Model 2: Shallow Low-Latency XGBoost (Optimized for real-time inference)
   - Model 3: Deep Regularized XGBoost (Complex non-linear interactions)
   - Model 4: XGBoost Learning-to-Rank (XGBRanker with pairwise query groups)
   - Model 5: Bayesian Hyperparameter-Optimized XGBoost (Optuna TPE)
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
import xgboost as xgb

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
# 1. Feature Engineering & Trip Ingestion
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
    df["query_id"] = df["trip_id"] + "_" + df["gps_index"].astype(str)
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
    Select candidate with highest predicted score per GPS observation
    and check if it is the ground-truth reference road.
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
        preds = (pos_scores >= 0.5).astype(int)
    else:  # XGBRanker or regression scoring
        pos_scores = model.predict(X_test)
        # Sigmoid calibration for log loss reporting
        calib_probs = 1.0 / (1.0 + np.exp(-np.clip(pos_scores, -20, 20)))
        loss_val = float(log_loss(y_test, np.column_stack([1 - calib_probs, calib_probs])))
        preds = (calib_probs >= 0.5).astype(int)
        
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
# 3. Model Suite: 5 Distinct XGBoost Architectures
# ==============================================================================

def run_xgboost_experiment(train_df: pd.DataFrame, test_df: pd.DataFrame, n_optuna_trials: int = 12) -> Dict:
    """Train and evaluate 5 XGBoost models across classifier, ranker, and Bayesian tuned."""
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["is_reference_candidate"].values.astype(int)
    groups_train = train_df["trip_id"].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_reference_candidate"].values.astype(int)
    
    # Calculate empirical class imbalance scale factor
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    imbalance_ratio = float(neg_count / max(1, pos_count))
    print(f"[*] Detected Class Imbalance Ratio: {imbalance_ratio:.2f} : 1")
    
    results = {}

    # --------------------------------------------------------------------------
    # Model 1: Baseline XGBoost (Standard Configuration)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 1: Baseline XGBoost (XGBClassifier)")
    print("="*70)
    m1 = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    t0 = time.time()
    m1.fit(X_train, y_train)
    m1_train_time = time.time() - t0
    
    m1_eval = evaluate_model(m1, X_test, y_test, test_df)
    m1_eval["Training_Time_s"] = m1_train_time
    m1_eval["Algorithm_Family"] = "Gradient Boosted Trees (XGBoost)"
    m1_eval["Configuration"] = f"n_est=100, depth=6, lr=0.1, scale_pos={imbalance_ratio:.1f}"
    results["Model 1 (Baseline XGBoost)"] = m1_eval
    print(f" [✓] Top-1 Acc: {m1_eval['Top1_Accuracy']:.2f}% | Macro F1: {m1_eval['Macro_F1']:.4f} | Log Loss: {m1_eval['Log_Loss']:.4f} | Train Time: {m1_train_time:.2f}s")

    # --------------------------------------------------------------------------
    # Model 2: Shallow Low-Latency XGBoost (Fast Inference)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 2: Shallow Low-Latency XGBoost (Ultra-Fast Inference)")
    print("="*70)
    m2 = xgb.XGBClassifier(
        n_estimators=45,
        max_depth=3,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    t0 = time.time()
    m2.fit(X_train, y_train)
    m2_train_time = time.time() - t0
    
    m2_eval = evaluate_model(m2, X_test, y_test, test_df)
    m2_eval["Training_Time_s"] = m2_train_time
    m2_eval["Algorithm_Family"] = "Gradient Boosted Trees (Low-Latency)"
    m2_eval["Configuration"] = f"n_est=45, depth=3, lr=0.15, subsample=0.8, colsample=0.8"
    results["Model 2 (Shallow Fast XGBoost)"] = m2_eval
    print(f" [✓] Top-1 Acc: {m2_eval['Top1_Accuracy']:.2f}% | Macro F1: {m2_eval['Macro_F1']:.4f} | Latency: {m2_eval['Latency_us_per_sample']:.1f} µs | Train Time: {m2_train_time:.2f}s")

    # --------------------------------------------------------------------------
    # Model 3: Deep Regularized XGBoost (Complex Non-Linear Representation)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 3: Deep Regularized XGBoost (High Complexity)")
    print("="*70)
    m3 = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=1.0,
        gamma=0.2,
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    t0 = time.time()
    m3.fit(X_train, y_train)
    m3_train_time = time.time() - t0
    
    m3_eval = evaluate_model(m3, X_test, y_test, test_df)
    m3_eval["Training_Time_s"] = m3_train_time
    m3_eval["Algorithm_Family"] = "Gradient Boosted Trees (Deep Regularized)"
    m3_eval["Configuration"] = "n_est=150, depth=8, lr=0.08, lambda=3.0, alpha=1.0, gamma=0.2"
    results["Model 3 (Deep Reg XGBoost)"] = m3_eval
    print(f" [✓] Top-1 Acc: {m3_eval['Top1_Accuracy']:.2f}% | Macro F1: {m3_eval['Macro_F1']:.4f} | Log Loss: {m3_eval['Log_Loss']:.4f} | Train Time: {m3_train_time:.2f}s")

    # --------------------------------------------------------------------------
    # Model 4: XGBoost Learning-to-Rank (XGBRanker with Pairwise Groups)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 4: XGBoost Learning-to-Rank (XGBRanker Pairwise)")
    print("="*70)
    # Group observations by query_id for ranking
    train_sorted = train_df.sort_values("query_id")
    X_rank_train = train_sorted[FEATURE_COLS].values
    y_rank_train = train_sorted["is_reference_candidate"].values.astype(int)
    qid_train = train_sorted["query_id"].values
    _, train_group_counts = np.unique(qid_train, return_counts=True)
    
    m4 = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=80,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    t0 = time.time()
    m4.fit(X_rank_train, y_rank_train, group=train_group_counts)
    m4_train_time = time.time() - t0
    
    m4_eval = evaluate_model(m4, X_test, y_test, test_df)
    m4_eval["Training_Time_s"] = m4_train_time
    m4_eval["Algorithm_Family"] = "Learning-to-Rank (XGBRanker)"
    m4_eval["Configuration"] = "objective=rank:pairwise, n_est=80, depth=5, lr=0.1"
    results["Model 4 (XGBRanker Pairwise)"] = m4_eval
    print(f" [✓] Top-1 Acc: {m4_eval['Top1_Accuracy']:.2f}% | Macro F1: {m4_eval['Macro_F1']:.4f} | Train Time: {m4_train_time:.2f}s")

    # --------------------------------------------------------------------------
    # Model 5: Bayesian Hyperparameter-Optimized XGBoost (Optuna TPE)
    # --------------------------------------------------------------------------
    print("\n" + "="*70)
    print(" [*] MODEL 5: BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna TPE)")
    print("="*70)
    gkf = GroupKFold(n_splits=3)
    
    # Representative subsample for fast cross-validation trials
    opt_size = min(len(X_train), 80000)
    opt_idx = np.random.choice(len(X_train), opt_size, replace=False) if len(X_train) > opt_size else np.arange(len(X_train))
    X_opt = X_train[opt_idx]
    y_opt = y_train[opt_idx]
    groups_opt = groups_train[opt_idx]
    
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 4, 8)
        lr = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
        n_est = trial.suggest_int("n_estimators", 50, 150, step=25)
        subsample = trial.suggest_float("subsample", 0.65, 0.95)
        colsample = trial.suggest_float("colsample_bytree", 0.65, 0.95)
        gamma = trial.suggest_float("gamma", 0.0, 3.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 0.01, 5.0, log=True)
        
        cv_f1s = []
        for tr_idx, val_idx in gkf.split(X_opt, y_opt, groups=groups_opt):
            clf = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=subsample,
                colsample_bytree=colsample,
                gamma=gamma,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                scale_pos_weight=imbalance_ratio,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            )
            clf.fit(X_opt[tr_idx], y_opt[tr_idx])
            val_preds = clf.predict(X_opt[val_idx])
            cv_f1s.append(f1_score(y_opt[val_idx], val_preds, average="macro"))
            
        return float(np.mean(cv_f1s))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    print(f"    Running {n_optuna_trials} Bayesian optimization trials using GroupKFold...")
    study.optimize(objective, n_trials=n_optuna_trials)
    best_params = study.best_params
    print(f"    [✓] Best Trial Score (CV Macro F1): {study.best_value:.4f}")
    print(f"    [✓] Optimal Hyperparameters: {best_params}")

    # Refit tuned model on complete training set
    m5 = xgb.XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        gamma=best_params["gamma"],
        reg_lambda=best_params["reg_lambda"],
        reg_alpha=best_params["reg_alpha"],
        scale_pos_weight=imbalance_ratio,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )
    t0 = time.time()
    m5.fit(X_train, y_train)
    m5_train_time = time.time() - t0
    
    m5_eval = evaluate_model(m5, X_test, y_test, test_df)
    m5_eval["Training_Time_s"] = m5_train_time
    m5_eval["Algorithm_Family"] = "Bayesian-Tuned XGBoost"
    m5_eval["Configuration"] = f"Optuna TPE: depth={best_params['max_depth']}, lr={best_params['learning_rate']:.3f}, n_est={best_params['n_estimators']}, lambda={best_params['reg_lambda']:.2f}"
    m5_eval["Best_Params"] = best_params
    m5_eval["CV_Macro_F1"] = float(study.best_value)
    results["Model 5 (Bayesian Tuned XGBoost)"] = m5_eval
    print(f" [✓] Tuned Top-1 Acc: {m5_eval['Top1_Accuracy']:.2f}% | Tuned Macro F1: {m5_eval['Macro_F1']:.4f} | Log Loss: {m5_eval['Log_Loss']:.4f}")

    # Export winning model artifact
    joblib.dump(m5, "xgboost_best_model.joblib")
    print(f"\n[*] Exported winning model artifact to: xgboost_best_model.joblib")

    return results


# ==============================================================================
# 4. Report Formatting & Section 3.1 Table
# ==============================================================================

def print_competition_table(results: Dict):
    """Format and print consolidated comparison table matching Section 3.1."""
    print("\n" + "="*95)
    print("SECTION 3.1: OVERALL MODEL COMPARISON (XGBOOST FAMILY EXPERIMENTS)")
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
    with open("xgboost_benchmark_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    print("[*] Benchmark results saved to: xgboost_benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(description="PathSnap XGBoost Training and Bayesian Optimization Pipeline")
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
    
    results = run_xgboost_experiment(train_df, test_df, n_optuna_trials=args.trials)
    print_competition_table(results)


if __name__ == "__main__":
    main()
