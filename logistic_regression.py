#!/usr/bin/env python3
"""
PathSnap candidate-ranking baseline using logistic regression.

One candidate row is one (GPS observation, road arc) pair.  The positive label
is is_reference_candidate from the processed candidate table.  Reference-only
columns are excluded from model inputs.  Dataset splits are by trip_id.

Outputs:
  logistic_results/metrics.json
  logistic_results/confusion_matrix.csv
  logistic_results/per_trip_metrics.csv
  logistic_results/predictions_train_validation.csv
  logistic_results/predictions_test.csv
  logistic_results/confusion_matrix.png
  logistic_results/roc_pr_placeholder.txt
"""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/neeraj-r-rugi/ML_Hackathon")
PROCESSED = ROOT / "processed"
TRAIN_PROCESSED = PROCESSED / "train"
TEST_PROCESSED = PROCESSED / "test"
RAW_ROOT = ROOT / "Student folder"
OUTPUT = ROOT / "logistic_results"
OUTPUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "distance_to_arc_m",
    "log_distance_to_arc_m",
    "projection_fraction",
    "length_m",
    "log_length_m",
    "bearing_sin",
    "bearing_cos",
    "candidate_rank",
    "candidate_count",
    "distance_rank_ratio",
    "distance_minus_min_m",
    "dt",
    "speed_mps",
    "speed_kmh",
    "heading_sin",
    "heading_cos",
    "heading_change_deg",
    "is_time_gap",
    "is_duplicate_time",
    "is_missing",
    "is_speed_outlier",
]


def read_trip_list(processed_split: str) -> list[str]:
    partition_dir = TRAIN_PROCESSED if processed_split == "train" else TEST_PROCESSED
    if not partition_dir.exists():
        raise FileNotFoundError(f"Processed partition missing: {partition_dir}")
    return sorted(p.name for p in partition_dir.iterdir() if p.is_dir())


def load_one_trip(split: str, trip_id: str) -> pd.DataFrame:
    partition_dir = TRAIN_PROCESSED if split == "train" else TEST_PROCESSED
    trip_dir = partition_dir / trip_id
    candidates = pd.read_csv(trip_dir / "candidate_features.csv")
    track = pd.read_csv(trip_dir / "track_features.csv")

    # Candidate-relative features require observations to be grouped by gps_index.
    valid = candidates["arc_id"].ge(0)
    candidate_counts = candidates.loc[valid].groupby("gps_index")["arc_id"].size()
    min_distance = candidates.loc[valid].groupby("gps_index")["distance_to_arc_m"].min()

    track_motion = track[[
        "gps_index", "dt", "speed_mps", "speed_kmh", "heading_deg",
        "is_time_gap", "is_duplicate_time", "is_missing", "is_speed_outlier",
    ]].copy()
    track_motion["heading_change_deg"] = (
        (track_motion["heading_deg"].diff() + 180.0) % 360.0
    ) - 180.0

    candidates = candidates.merge(
        track_motion,
        on="gps_index",
        how="left",
    )

    if "is_reference_candidate" not in candidates:
        route_path = RAW_ROOT / ("train" if split == "train" else "student_test") / trip_id / f"{trip_id}.route"
        if route_path.exists():
            route = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])
            route_arc_ids = set(route["arc_id"].astype(int).tolist())
            candidates["is_reference_route_arc"] = candidates["arc_id"].isin(route_arc_ids)

            valid_route_candidates = candidates[
                candidates["is_reference_route_arc"] & candidates["arc_id"].ge(0)
            ]
            if valid_route_candidates.empty:
                labels = pd.DataFrame(columns=["gps_index", "reference_arc_id", "reference_distance_m"])
            else:
                nearest = valid_route_candidates.sort_values(
                    ["gps_index", "distance_to_arc_m"]
                ).drop_duplicates("gps_index")
                labels = nearest[["gps_index", "arc_id", "distance_to_arc_m"]].rename(columns={
                    "arc_id": "reference_arc_id",
                    "distance_to_arc_m": "reference_distance_m",
                })

            candidates = candidates.merge(labels, on="gps_index", how="left")
            candidates["reference_arc_id"] = candidates["reference_arc_id"].fillna(-1).astype(int)
            candidates["is_reference_candidate"] = (
                candidates["arc_id"].ge(0)
                & candidates["arc_id"].eq(candidates["reference_arc_id"])
            ).astype(int)

    candidates["trip_id"] = trip_id
    candidates["split"] = split
    candidates["candidate_count"] = candidates["gps_index"].map(candidate_counts).fillna(0).astype(float)
    candidates["distance_rank_ratio"] = candidates["candidate_rank"] / candidates["candidate_count"].clip(lower=1)
    candidates["distance_minus_min_m"] = (
        candidates["distance_to_arc_m"]
        - candidates["gps_index"].map(min_distance)
    ).fillna(0)

    candidates["log_distance_to_arc_m"] = np.log1p(candidates["distance_to_arc_m"].clip(lower=0))
    candidates["log_length_m"] = np.log1p(candidates["length_m"].clip(lower=0))

    bearing = np.deg2rad(candidates["bearing_deg"].fillna(0).to_numpy())
    heading = np.deg2rad(candidates["heading_deg"].fillna(0).to_numpy())
    heading_change = np.deg2rad(candidates["heading_change_deg"].fillna(0).to_numpy())
    candidates["bearing_sin"] = np.sin(bearing)
    candidates["bearing_cos"] = np.cos(bearing)
    candidates["heading_sin"] = np.sin(heading)
    candidates["heading_cos"] = np.cos(heading)

    for column in [
        "is_time_gap", "is_duplicate_time", "is_missing", "is_speed_outlier"
    ]:
        candidates[column] = candidates[column].fillna(False).astype(int)

    return candidates


def fuse_trip_tables(splits: Iterable[str]) -> pd.DataFrame:
    frames = []
    for split in splits:
        for trip_id in read_trip_list(split):
            frame = load_one_trip(split, trip_id)
            if "is_reference_candidate" not in frame:
                raise RuntimeError(f"Missing training label in {split}/{trip_id}")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def make_trip_split(trip_ids: list[str], seed: int = RANDOM_STATE) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(trip_ids), dtype=str)
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * 0.20)))
    validation = set(shuffled[:validation_count].tolist())
    train = set(shuffled[validation_count:].tolist())
    return train, validation


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
        "candidate_rows": int(len(y_true)),
        "positive_rows": int(np.sum(y_true)),
    }


def top1_metrics(predictions: pd.DataFrame) -> dict:
    """Observation-level top-1 and candidate-miss metrics."""
    valid = predictions["arc_id"].ge(0)
    groups = predictions.loc[valid].groupby(["split", "trip_id", "gps_index"], sort=False)

    top1_rows = []
    miss_rows = []

    for (split, trip_id, gps_index), group in groups:
        if "reference_arc_id" not in group:
            continue
        reference = group["reference_arc_id"].dropna()
        if reference.empty:
            continue
        reference_arc = int(reference.iloc[0])
        if reference_arc < 0:
            continue

        if reference_arc not in set(group["arc_id"].astype(int)):
            miss_rows.append({"split": split, "trip_id": trip_id, "gps_index": int(gps_index)})
            continue

        winner = group.sort_values("probability", ascending=False).iloc[0]
        top1_rows.append({
            "split": split,
            "trip_id": trip_id,
            "gps_index": int(gps_index),
            "predicted_arc": int(winner["arc_id"]),
            "reference_arc": reference_arc,
            "top1_correct": int(winner["arc_id"] == reference_arc),
        })

    top1 = pd.DataFrame(top1_rows)
    miss = pd.DataFrame(miss_rows)

    eligible = len(top1)
    accuracy = float(top1.top1_correct.mean()) if eligible else 0.0

    eligible_plus_misses = eligible + len(miss)
    candidate_miss_rate = float(len(miss) / eligible_plus_misses) if eligible_plus_misses else 0.0

    return {
        "observation_level_top1_accuracy": accuracy,
        "candidate_miss_rate": candidate_miss_rate,
        "eligible_observations": eligible,
        "candidate_missed_observations": len(miss),
    }, top1, miss


def route_continuity_and_sequence_metrics(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Compute arc-transition continuity and route overlap for top-1 predictions.

    Route continuity: adjacent top-1 arcs are considered continuous if they are
    the same arc or the next arc starts at the current arc's destination node.

    Sequence performance: compare collapsed top-1 arc sequences to collapsed
    per-observation reference arcs using Jaccard and longest-common-subsequence
    ratios.
    """
    per_trip = []

    for (split, trip_id), group in predictions.groupby(["split", "trip_id"], sort=True):
        winners = (group.groupby("gps_index", sort=True)
                   .apply(lambda g: g.sort_values("probability", ascending=False).iloc[0], include_groups=False)
                   .reset_index())

        predicted = winners["arc_id"].astype(int).tolist()
        reference_arc = winners["reference_arc_id"].fillna(-1).astype(int).tolist()
        reference = [arc for arc in reference_arc if arc >= 0]

        predicted_collapsed = []
        for arc in predicted:
            if arc >= 0 and (not predicted_collapsed or predicted_collapsed[-1] != arc):
                predicted_collapsed.append(arc)

        raw_route = RAW_ROOT / ("train" if split == "train" else "student_test") / trip_id / f"{trip_id}.route"
        reference_route = []
        if raw_route.exists():
            reference_route = pd.read_csv(raw_route, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].tolist()
            collapsed_reference = []
            for arc in reference_route:
                if not collapsed_reference or collapsed_reference[-1] != arc:
                    collapsed_reference.append(int(arc))
            reference_route = collapsed_reference

        # Directed-arc continuity from the trip-local arcs file.
        arcs_path = RAW_ROOT / ("train" if split == "train" else "student_test") / trip_id / f"{trip_id}.arcs"
        arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None,
                           names=["from_node", "to_node"])
        current_continuous = 0
        adjacent_pairs = 0
        for left, right in zip(predicted, predicted[1:]):
            if left < 0 or right < 0:
                continue
            adjacent_pairs += 1
            if left == right:
                current_continuous += 1
            elif 0 <= left < len(arcs) and 0 <= right < len(arcs):
                if int(arcs.iloc[left].to_node) == int(arcs.iloc[right].from_node):
                    current_continuous += 1

        continuity = current_continuous / adjacent_pairs if adjacent_pairs else 1.0

        def jaccard(a: list[int], b: list[int]) -> float:
            sa, sb = set(a), set(b)
            return len(sa & sb) / len(sa | sb) if sa or sb else 1.0

        def lcs_ratio(a: list[int], b: list[int]) -> float:
            if not a or not b:
                return 1.0 if not a and not b else 0.0
            previous = [0] * (len(b) + 1)
            for i in range(1, len(a) + 1):
                current = [0] * (len(b) + 1)
                for j in range(1, len(b) + 1):
                    if a[i - 1] == b[j - 1]:
                        current[j] = previous[j - 1] + 1
                    else:
                        current[j] = max(previous[j], current[j - 1])
                previous = current
            lcs = previous[-1]
            return lcs / max(len(a), len(b))

        per_trip.append({
            "split": split,
            "trip_id": trip_id,
            "route_continuity": continuity,
            "route_jaccard_collapsed": jaccard(predicted_collapsed, reference_route),
            "route_lcs_ratio_collapsed": lcs_ratio(predicted_collapsed, reference_route),
            "predicted_collapsed_arcs": len(predicted_collapsed),
            "reference_collapsed_arcs": len(reference_route),
        })

    table = pd.DataFrame(per_trip)
    aggregate = {
        "route_continuity": float(table.route_continuity.mean()) if len(table) else 0.0,
        "sequence_jaccard_mean": float(table.route_jaccard_collapsed.mean()) if len(table) else 0.0,
        "sequence_lcs_ratio_mean": float(table.route_lcs_ratio_collapsed.mean()) if len(table) else 0.0,
    }
    return aggregate, table


def plot_confusion(confusion: dict, title: str, path: Path) -> None:
    matrix = np.array([
        [confusion["tn"], confusion["fp"]],
        [confusion["fn"], confusion["tp"]],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["negative", "positive"])
    ax.set_yticks([0, 1], ["negative", "positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading and fusing train trip tables...")
    train_all = fuse_trip_tables(["train"])

    trip_ids = sorted(train_all.trip_id.unique().tolist())
    train_ids, validation_ids = make_trip_split(trip_ids)
    train_frame = train_all[train_all.trip_id.isin(train_ids)].copy()
    validation_frame = train_all[train_all.trip_id.isin(validation_ids)].copy()

    X_train = train_frame[FEATURE_COLUMNS]
    y_train = train_frame["is_reference_candidate"].fillna(False).astype(int).to_numpy()
    X_validation = validation_frame[FEATURE_COLUMNS]
    y_validation = validation_frame["is_reference_candidate"].fillna(False).astype(int).to_numpy()

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=300,
            C=1.0,
            random_state=RANDOM_STATE,
        )),
    ])

    print("Training logistic regression...")
    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - training_start

    inference_start = time.perf_counter()
    validation_probability = model.predict_proba(X_validation)[:, 1]
    validation_inference_seconds = time.perf_counter() - inference_start

    validation_metrics = binary_metrics(y_validation, validation_probability)

    validation_predictions = validation_frame[[
        "split", "trip_id", "gps_index", "arc_id", "reference_arc_id"
    ]].copy()
    validation_predictions["probability"] = validation_probability
    validation_predictions.to_csv(OUTPUT / "predictions_train_validation.csv", index=False)

    top1_validation, top1_table_validation, candidate_misses_validation = top1_metrics(validation_predictions)
    sequence_validation, sequence_table_validation = route_continuity_and_sequence_metrics(validation_predictions)

    print("Loading and evaluating test partition...")
    test_frame = fuse_trip_tables(["test"])
    X_test = test_frame[FEATURE_COLUMNS]
    y_test = test_frame["is_reference_candidate"].fillna(False).astype(int).to_numpy()

    test_inference_start = time.perf_counter()
    test_probability = model.predict_proba(X_test)[:, 1]
    test_inference_seconds = time.perf_counter() - test_inference_start

    test_metrics = binary_metrics(y_test, test_probability)

    test_predictions = test_frame[[
        "split", "trip_id", "gps_index", "arc_id", "reference_arc_id"
    ]].copy()
    test_predictions["probability"] = test_probability
    test_predictions.to_csv(OUTPUT / "predictions_test.csv", index=False)

    top1_test, top1_table_test, candidate_misses_test = top1_metrics(test_predictions)
    sequence_test, sequence_table_test = route_continuity_and_sequence_metrics(test_predictions)

    candidate_misses_validation.to_csv(OUTPUT / "candidate_misses_validation.csv", index=False)
    candidate_misses_test.to_csv(OUTPUT / "candidate_misses_test.csv", index=False)
    sequence_table_validation.to_csv(OUTPUT / "sequence_metrics_validation.csv", index=False)
    sequence_table_test.to_csv(OUTPUT / "sequence_metrics_test.csv", index=False)

    all_metrics = {
        "model": "LogisticRegression",
        "train_source": str(TRAIN_PROCESSED),
        "test_source": str(TEST_PROCESSED),
        "primary_metric": "observation_level_top1_accuracy",
        "primary_metric_reason": (
            "The route-matching objective selects one road arc for each GPS observation. "
            "Top-1 accuracy directly measures this observation-level decision and is more "
            "appropriate than candidate-row accuracy, which rewards models for predicting "
            "many negative candidate rows and may ignore the winning road choice."
        ),
        "random_state": RANDOM_STATE,
        "train_trip_count": len(train_ids),
        "validation_trip_count": len(validation_ids),
        "test_trip_count": len(test_frame.trip_id.unique()),
        "train_trip_ids": sorted(train_ids),
        "validation_trip_ids": sorted(validation_ids),
        "training_time_seconds": training_seconds,
        "validation_inference_time_seconds": validation_inference_seconds,
        "test_inference_time_seconds": test_inference_seconds,
        "validation_binary_metrics": validation_metrics,
        "validation_top1_and_miss": top1_validation,
        "validation_sequence_metrics": sequence_validation,
        "test_binary_metrics": test_metrics,
        "test_top1_and_miss": top1_test,
        "test_sequence_metrics": sequence_test,
        "features": FEATURE_COLUMNS,
    }

    (OUTPUT / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    confusion = validation_metrics["confusion_matrix"]
    rows = pd.DataFrame({
        "actual_negative": [confusion["tn"], confusion["fn"]],
        "actual_positive": [confusion["fp"], confusion["tp"]],
    }, index=["predicted_negative", "predicted_positive"])
    rows.to_csv(OUTPUT / "confusion_matrix.csv")
    plot_confusion(confusion, "Logistic Regression - Validation Confusion Matrix",
                   OUTPUT / "confusion_matrix.png")

    print(json.dumps(all_metrics, indent=2))
    print("Results written to:", OUTPUT)


if __name__ == "__main__":
    main()
