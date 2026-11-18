#!/usr/bin/env python3
"""
PathSnap: Laptop 1 — Backend Model & API Server
================================================
PES University ML Hackathon: Distributed GPS Map-Matching System

Responsibilities (Laptop 1):
----------------------------
1. Hosts all trained ML models:
   - XGBoost (Gradient Boosting / Bayesian Tuned)
   - Random Forest (Bagging Ensemble)
   - Logistic Regression (L2 / L-BFGS)
   - Support Vector Machine (SGD Modified Huber / Linear)
   - K-Nearest Neighbors (KD-Tree Spatial Index)
2. Supports both the new Evaluator Test Set (evaluator_test) and Student Test Set (student_test).
3. Executes HMM Viterbi road network graph decoding with dynamic Dijkstra memoization and multiple optimizer profiles.
4. Computes server-side Top-1 Accuracy, route continuity, and latency profiles.
5. High-visibility terminal logging for live demo evaluation.
"""

import os
import sys
import time
import socket
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

from hmm_decoder import HMMViterbiMapMatcher, DEFAULT_FEATURE_COLS
from enhanced_preprocessing import process_trip_enhanced

# Global State
MODELS: Dict[str, Any] = {}
DATA_ROOT = Path(".")
PROCESSED_ROOT = Path("processed_enhanced")

MODEL_CONFIGS = {
    "xgboost": ("model_xgboost.joblib", "XGBoost (Bayesian Tuned / Depth 5)"),
    "random_forest": ("model_random_forest.joblib", "Random Forest (100 Trees / Depth 14)"),
    "logistic_regression": ("model_logistic_regression.joblib", "Logistic Regression (L2 / L-BFGS)"),
    "svm": ("model_svm.joblib", "Support Vector Machine (SGD Modified Huber)"),
    "knn": ("model_knn.joblib", "K-Nearest Neighbors (KD-Tree Spatial Index)")
}


def get_local_ips() -> List[str]:
    """Retrieve local IPv4 address quickly without blocking hostname DNS."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        outbound = s.getsockname()[0]
        s.close()
        if outbound and not outbound.startswith("127."):
            ips.append(outbound)
    except Exception:
        pass
    if "127.0.0.1" not in ips:
        ips.append("127.0.0.1")
    return ips


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts into memory once at startup."""
    print("\n" + "=" * 80)
    print(" 🚀 PATHSNAP DISTRIBUTED MAP-MATCHING SYSTEM — LAPTOP 1 API SERVER")
    print("=" * 80)

    for m_key, (m_file, m_desc) in MODEL_CONFIGS.items():
        p = Path(m_file)
        if p.exists():
            MODELS[m_key] = joblib.load(p)
            print(f" [✓] Loaded {m_desc:<45} : {m_file}")
        else:
            print(f" [!] Warning: {m_file} not found!")

    ips = get_local_ips()
    print("\n [★] LAPTOP 2 CONNECTION INSTRUCTIONS:")
    print("     Configure Laptop 2 Evaluator UI with either of these URLs:")
    for ip in ips:
        print(f"     👉  http://{ip}:8000")
    print("=" * 80 + "\n")
    yield


app = FastAPI(
    title="PathSnap GPS Map-Matching API",
    description="PES University ML Hackathon: Distributed GPS Map-Matching API (Laptop 1 Backend)",
    version="2.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    trip_id: str
    model: str = "xgboost"
    split: Optional[str] = None  # "evaluator_test" or "student_test"
    optimizer: str = "standard_hmm"  # standard_hmm, highway_hmm, urban_hmm, custom_hmm, none
    sigma_z: Optional[float] = 8.0
    beta: Optional[float] = 15.0
    wrong_way_penalty: Optional[float] = 10.0


class CompareRequest(BaseModel):
    trip_id: str
    split: Optional[str] = None
    optimizer: str = "standard_hmm"
    sigma_z: Optional[float] = 8.0
    beta: Optional[float] = 15.0
    wrong_way_penalty: Optional[float] = 10.0


def resolve_trip_dir(trip_id: str) -> Path:
    """Locates preprocessed trip directory or processes on the fly."""
    candidate_locations = [
        PROCESSED_ROOT / "evaluator_test" / trip_id,
        PROCESSED_ROOT / "student_test" / trip_id,
        PROCESSED_ROOT / "train" / trip_id
    ]

    for loc in candidate_locations:
        if (loc / "candidate_features.csv").exists():
            return loc

    # If not preprocessed, process on the fly
    if Path(f"evaluator_test/{trip_id}").exists():
        print(f" [*] Preprocessing evaluator trip {trip_id} on the fly...")
        process_trip_enhanced(trip_id, "evaluator_test", Path("."), PROCESSED_ROOT)
        return PROCESSED_ROOT / "evaluator_test" / trip_id
    elif Path(f"Student folder/student_test/{trip_id}").exists():
        print(f" [*] Preprocessing student test trip {trip_id} on the fly...")
        process_trip_enhanced(trip_id, "student_test", Path("Student folder"), PROCESSED_ROOT)
        return PROCESSED_ROOT / "student_test" / trip_id
    else:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} data not found on server.")


def get_matcher_for_optimizer(optimizer: str, sigma_z: Optional[float], beta: Optional[float], wrong_way_penalty: Optional[float]):
    opt = (optimizer or "standard_hmm").lower().strip()
    if opt == "highway_hmm":
        sz = 6.0
        b = 25.0
        ww = 15.0
    elif opt == "urban_hmm":
        sz = 12.0
        b = 10.0
        ww = 8.0
    elif opt == "custom_hmm":
        sz = float(sigma_z if sigma_z is not None else 8.0)
        b = float(beta if beta is not None else 15.0)
        ww = float(wrong_way_penalty if wrong_way_penalty is not None else 10.0)
    else:  # standard_hmm or fallback
        sz = 8.0
        b = 15.0
        ww = 10.0

    return HMMViterbiMapMatcher(
        sigma_z=sz,
        beta=b,
        wrong_way_penalty=ww
    ), (opt == "none")


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint displaying server and model status."""
    return {
        "status": "online",
        "service": "PathSnap Map-Matching API (Laptop 1 Backend)",
        "available_models": list(MODELS.keys()),
        "available_optimizers": [
            "standard_hmm",
            "highway_hmm",
            "urban_hmm",
            "custom_hmm",
            "none"
        ],
        "local_ips": get_local_ips(),
        "timestamp": time.time()
    }


@app.get("/trips")
def list_evaluation_trips():
    """Returns available evaluation trips from both evaluator_test and student_test."""
    all_trips = []

    # 1. Official Evaluator Test Set (Judges zip)
    eval_dir = Path("evaluator_test")
    if eval_dir.exists():
        for p in sorted(eval_dir.glob("*")):
            if p.is_dir() and p.name.isdigit():
                tid = p.name
                track_file = p / f"{tid}.track"
                n_pts = sum(1 for _ in open(track_file)) if track_file.exists() else 0
                all_trips.append({
                    "trip_id": tid,
                    "split": "evaluator_test",
                    "label": f"Trip {tid} (Evaluator Test Set - {n_pts:,} pts)",
                    "gps_points_count": n_pts,
                    "ground_truth_available": (p / f"{tid}.route").exists()
                })

    # 2. Student Test Set
    stud_dir = Path("Student folder/student_test")
    if stud_dir.exists():
        for p in sorted(stud_dir.glob("*")):
            if p.is_dir() and p.name.isdigit():
                tid = p.name
                track_file = p / f"{tid}.track"
                n_pts = sum(1 for _ in open(track_file)) if track_file.exists() else 0
                all_trips.append({
                    "trip_id": tid,
                    "split": "student_test",
                    "label": f"Trip {tid} (Student Test Set - {n_pts:,} pts)",
                    "gps_points_count": n_pts,
                    "ground_truth_available": (p / f"{tid}.route").exists()
                })

    return {"status": "success", "trips": all_trips}


@app.post("/predict")
def predict_trip(req: PredictRequest):
    """Processes the selected trip through candidate scoring and chosen optimizer."""
    t_start = time.time()
    trip_id = req.trip_id.strip()
    model_name = req.model.lower().strip()

    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Choose from {list(MODELS.keys())}"
        )

    classifier = MODELS[model_name]
    trip_processed = resolve_trip_dir(trip_id)

    # Ingest Features
    cands_df = pd.read_csv(trip_processed / "candidate_features.csv")
    arcs_df = pd.read_csv(trip_processed / "arcs_features.csv")
    track_df = pd.read_csv(trip_processed / "track_features.csv")
    nodes_df = pd.read_csv(trip_processed / "nodes_features.csv")

    matcher, is_no_optimizer = get_matcher_for_optimizer(
        req.optimizer, req.sigma_z, req.beta, req.wrong_way_penalty
    )

    print("\n" + "-" * 75)
    print(f" 📥 [EVALUATION REQUEST]")
    print(f"    Trip ID   : {trip_id} ({trip_processed.parent.name})")
    print(f"    Model     : {MODEL_CONFIGS[model_name][1]}")
    print(f"    Optimizer : {req.optimizer.upper()} (σ={matcher.sigma_z}m, β={matcher.beta}m)")
    print(f"    Points    : {len(track_df):,} GPS points | {len(cands_df):,} candidates")
    print("-" * 75)

    # Decode via HMM Viterbi Map Matcher
    res = matcher.decode(
        cands_df=cands_df,
        arcs_df=arcs_df,
        track_df=track_df,
        classifier=classifier,
        feature_cols=DEFAULT_FEATURE_COLS,
        trip_id=trip_id
    )

    # If user selected NO optimizer, override viterbi sequence with pure classifier sequence
    if is_no_optimizer:
        res.viterbi_arc_sequence = res.classifier_arc_sequence
        res.viterbi_top1_accuracy = res.classifier_top1_accuracy
        res.accuracy_gain = 0.0

    # Build Pointwise Preview
    preview_points = []
    node_map = dict(zip(nodes_df["node_id"], zip(nodes_df["lat"], nodes_df["lon"])))
    arc_to_endpoints = dict(zip(arcs_df["arc_id"], zip(arcs_df["from_node"], arcs_df["to_node"])))

    step_sample = max(1, len(track_df) // 30)
    for i in range(0, len(track_df), step_sample):
        pt_row = track_df.iloc[i]
        gps_idx = int(pt_row["gps_index"])
        clf_arc = res.classifier_arc_sequence[i] if i < len(res.classifier_arc_sequence) else -1
        vit_arc = res.viterbi_arc_sequence[i] if i < len(res.viterbi_arc_sequence) else -1
        gt_arc = res.ground_truth_arc_sequence[i] if res.ground_truth_arc_sequence and i < len(res.ground_truth_arc_sequence) else -1

        preview_points.append({
            "point_index": gps_idx,
            "latitude": float(pt_row["lat"]),
            "longitude": float(pt_row["lon"]),
            "speed_kmh": round(float(pt_row["speed_kmh"]), 1),
            "classifier_arc_id": int(clf_arc),
            "viterbi_arc_id": int(vit_arc),
            "ground_truth_arc_id": int(gt_arc),
            "viterbi_correct": bool(vit_arc == gt_arc) if gt_arc != -1 else None
        })

    # Build Map Geometries
    gps_coords = [[float(r.lat), float(r.lon)] for r in track_df.itertuples()]

    def arcs_to_linestrings(arc_list: List[int]) -> List[List[List[float]]]:
        lines = []
        for arc_id in arc_list:
            if arc_id in arc_to_endpoints:
                u, v = arc_to_endpoints[arc_id]
                if u in node_map and v in node_map:
                    lines.append([list(node_map[u]), list(node_map[v])])
        return lines

    viterbi_lines = arcs_to_linestrings(res.viterbi_arc_sequence)
    gt_lines = arcs_to_linestrings(res.ground_truth_arc_sequence) if res.ground_truth_arc_sequence else []

    total_req_time_ms = (time.time() - t_start) * 1000.0

    # High-Visibility Terminal Output
    clf_str = f"{res.classifier_top1_accuracy:.2f}%" if res.classifier_top1_accuracy is not None else "N/A"
    vit_str = f"{res.viterbi_top1_accuracy:.2f}%" if res.viterbi_top1_accuracy is not None else "N/A"
    gain_str = f"+{res.accuracy_gain:.2f}%" if res.accuracy_gain is not None else "N/A"

    print(f" [✓] Completed Evaluation for Trip {trip_id}:")
    print(f"     • Classifier Top-1 Accuracy : {clf_str}")
    print(f"     • HMM Viterbi Top-1 Accuracy: {vit_str}")
    print(f"     • Net Improvement           : {gain_str} 🚀")
    print(f"     • Route Continuity          : {res.continuity_score_percent:.2f}%")
    print(f"     • Latency Total             : {total_req_time_ms:.1f} ms (Vit: {res.viterbi_latency_ms:.1f}ms)")
    print("-" * 75 + "\n")

    return {
        "status": "success",
        "trip_id": trip_id,
        "split": trip_processed.parent.name,
        "model_used": model_name,
        "model_display_name": MODEL_CONFIGS[model_name][1],
        "optimizer_used": req.optimizer,
        "optimizer_params": {"sigma_z": matcher.sigma_z, "beta": matcher.beta},
        "num_gps_points": res.num_gps_points,
        "classifier_accuracy_percent": round(res.classifier_top1_accuracy, 2) if res.classifier_top1_accuracy is not None else None,
        "viterbi_accuracy_percent": round(res.viterbi_top1_accuracy, 2) if res.viterbi_top1_accuracy is not None else None,
        "accuracy_gain_percent": round(res.accuracy_gain, 2) if res.accuracy_gain is not None else None,
        "continuity_score_percent": round(res.continuity_score_percent, 2),
        "timing_ms": {
            "feature_latency_ms": round(res.feature_latency_ms, 2),
            "inference_latency_ms": round(res.inference_latency_ms, 2),
            "viterbi_latency_ms": round(res.viterbi_latency_ms, 2),
            "total_latency_ms": round(total_req_time_ms, 2)
        },
        "preview_table": preview_points,
        "map_data": {
            "center": [float(track_df["lat"].mean()), float(track_df["lon"].mean())],
            "gps_coords": gps_coords[::2],
            "viterbi_lines": viterbi_lines,
            "ground_truth_lines": gt_lines
        }
    }


@app.post("/compare_all")
def compare_all_models(req: CompareRequest):
    """Benchmarks all 5 models on the selected trip and returns comparative leaderboard."""
    trip_id = req.trip_id.strip()
    trip_processed = resolve_trip_dir(trip_id)

    cands_df = pd.read_csv(trip_processed / "candidate_features.csv")
    arcs_df = pd.read_csv(trip_processed / "arcs_features.csv")
    track_df = pd.read_csv(trip_processed / "track_features.csv")

    matcher, is_no_optimizer = get_matcher_for_optimizer(
        req.optimizer, req.sigma_z, req.beta, req.wrong_way_penalty
    )

    results = []
    for m_key in ["xgboost", "random_forest", "logistic_regression", "svm", "knn"]:
        if m_key not in MODELS:
            continue
        clf = MODELS[m_key]
        t0 = time.time()
        res = matcher.decode(
            cands_df=cands_df,
            arcs_df=arcs_df,
            track_df=track_df,
            classifier=clf,
            feature_cols=DEFAULT_FEATURE_COLS,
            trip_id=trip_id
        )
        total_lat = (time.time() - t0) * 1000.0

        if is_no_optimizer:
            vit_acc = res.classifier_top1_accuracy
            gain = 0.0
        else:
            vit_acc = res.viterbi_top1_accuracy
            gain = res.accuracy_gain

        results.append({
            "model_key": m_key,
            "model_name": MODEL_CONFIGS[m_key][1],
            "classifier_accuracy_percent": round(res.classifier_top1_accuracy, 2) if res.classifier_top1_accuracy is not None else None,
            "viterbi_accuracy_percent": round(vit_acc, 2) if vit_acc is not None else None,
            "accuracy_gain_percent": round(gain, 2) if gain is not None else None,
            "continuity_score_percent": round(res.continuity_score_percent, 2),
            "latency_ms": round(total_lat, 1)
        })

    # Sort by viterbi accuracy descending
    results.sort(key=lambda x: (x["viterbi_accuracy_percent"] or 0.0), reverse=True)

    return {
        "status": "success",
        "trip_id": trip_id,
        "split": trip_processed.parent.name,
        "optimizer_used": req.optimizer,
        "leaderboard": results
    }


def main():
    parser = argparse.ArgumentParser(description="PathSnap Laptop 1 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000)")
    args = parser.parse_args()

    uvicorn.run("app:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
