#!/usr/bin/env python3
"""
PathSnap: Hidden Markov Model (HMM) & Viterbi Sequence Decoder
==============================================================
Complies with Hackathon Section 5 (HMM and Viterbi Route Reconstruction)
and Models & Training Guidelines Section 4:

1. Hidden States: Candidate directed road arcs within the search radius for each GPS point.
2. Emission Probabilities: P(z_t | s_i^t) combines the ML classifier score (e.g. XGBoost/SVM)
   with spatial perpendicular distance likelihood N(0, sigma_z^2) and heading orientation penalty.
3. Transition Probabilities: P(s_j^{t+1} | s_i^t) models network distance discrepancy
   relative to GPS Euclidean step displacement using the exponential error distribution:
   P = (1/beta) * exp(-|d_net - d_gps| / beta) with Dijkstra shortest path on the road graph.
4. Sequence Decoding: Dynamic programming Viterbi recurrence in log-space with backtracking.
5. Topological Continuity: Evaluates the proportion of consecutive arc transitions that
   are topologically connected in the directed road graph.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx

DEFAULT_FEATURE_COLS = [
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


@dataclass
class MatchResult:
    trip_id: str
    num_gps_points: int
    classifier_top1_accuracy: Optional[float]
    viterbi_top1_accuracy: Optional[float]
    accuracy_gain: Optional[float]
    continuity_score_percent: float
    classifier_arc_sequence: List[int]
    viterbi_arc_sequence: List[int]
    ground_truth_arc_sequence: Optional[List[int]]
    feature_latency_ms: float
    inference_latency_ms: float
    viterbi_latency_ms: float
    total_latency_ms: float


class HMMViterbiMapMatcher:
    """
    High-performance Map-Matching Sequence Decoder combining Machine Learning
    emission probabilities with Road Network Graph Viterbi Decoding.
    """

    def __init__(
        self,
        sigma_z: float = 8.0,
        beta: float = 15.0,
        wrong_way_penalty: float = 10.0,
        disconnected_penalty: float = 20.0,
        min_cutoff_m: float = 120.0
    ):
        """
        Parameters:
        -----------
        sigma_z : float
            Standard deviation of GPS measurement error in meters (Gaussian emission model).
        beta : float
            Scale parameter for network vs GPS distance discrepancy (Exponential transition model).
        wrong_way_penalty : float
            Log-space penalty deducted for candidates moving opposite to road arc bearing.
        disconnected_penalty : float
            Log-space penalty deducted when no directed graph path connects consecutive arcs.
        min_cutoff_m : float
            Minimum search horizon in meters for single-source Dijkstra path calculation.
        """
        self.sigma_z = sigma_z
        self.beta = beta
        self.wrong_way_penalty = wrong_way_penalty
        self.disconnected_penalty = disconnected_penalty
        self.min_cutoff_m = min_cutoff_m

    def build_graph(self, arcs_df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict[int, Tuple[int, int, float]], Dict[int, Dict[int, float]]]:
        """
        Construct directed graph and fast O(1) adjacency structures.
        """
        G = nx.DiGraph()
        arc_lookup = {}
        for r in arcs_df.itertuples():
            u = int(r.from_node)
            v = int(r.to_node)
            length = float(r.length_m)
            arc_id = int(r.arc_id)
            G.add_edge(u, v, weight=length)
            arc_lookup[arc_id] = (u, v, length)

        adj_weights = {u: {nbr: data["weight"] for nbr, data in nbrs.items()} for u, nbrs in G.adjacency()}
        return G, arc_lookup, adj_weights

    def evaluate_continuity(
        self, 
        arc_seq: List[int], 
        G: nx.DiGraph, 
        arc_lookup: Dict[int, Tuple[int, int, float]]
    ) -> float:
        """
        Calculate route continuity: percentage of consecutive arc pairs that are
        either identical or connected via a valid directed path in the road graph.
        """
        if len(arc_seq) <= 1:
            return 100.0

        valid_transitions = 0
        total_transitions = len(arc_seq) - 1

        for t in range(total_transitions):
            a1, a2 = arc_seq[t], arc_seq[t + 1]
            if a1 == a2:
                valid_transitions += 1
                continue

            u1, v1, _ = arc_lookup.get(a1, (-1, -1, 0.0))
            u2, v2, _ = arc_lookup.get(a2, (-1, -1, 0.0))
            if v1 == u2:
                valid_transitions += 1
            elif v1 >= 0 and u2 >= 0 and nx.has_path(G, v1, u2):
                valid_transitions += 1

        return (valid_transitions / total_transitions) * 100.0

    def decode(
        self,
        cands_df: pd.DataFrame,
        arcs_df: pd.DataFrame,
        track_df: pd.DataFrame,
        classifier: Any,
        feature_cols: Optional[List[str]] = None,
        trip_id: str = "unknown"
    ) -> MatchResult:
        """
        Execute complete ML scoring + HMM Viterbi decoding pipeline.
        """
        t_total_start = time.time()
        feature_cols = feature_cols or DEFAULT_FEATURE_COLS

        # 1. Feature Preprocessing & Hygiene
        t_feat_start = time.time()
        df = cands_df.copy()
        if "log_distance" not in df.columns:
            df["log_distance"] = np.log1p(np.maximum(df["distance_to_arc_m"].fillna(0.0), 0.0))
        if "log_length" not in df.columns:
            df["log_length"] = np.log1p(np.maximum(df["length_m"].fillna(0.0), 0.0))
        if "dist_to_length_ratio" not in df.columns:
            df["dist_to_length_ratio"] = df["dist_to_length_ratio"].fillna(0.0)

        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # Align with classifier expected feature count
        n_expected = getattr(classifier, "n_features_in_", len(feature_cols))
        used_cols = feature_cols[:n_expected]
        X = df[used_cols].values
        feat_lat_ms = (time.time() - t_feat_start) * 1000.0

        # 2. Classifier Emission Scoring
        t_inf_start = time.time()
        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(X)[:, 1]
        elif hasattr(classifier, "decision_function"):
            scores = classifier.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))
        else:
            scores = classifier.predict(X)
            probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))

        df["clf_prob"] = probs
        inf_lat_ms = (time.time() - t_inf_start) * 1000.0

        # 3. Classifier-Only Baseline Predictions
        cands_sorted = df.sort_values(["gps_index", "clf_prob"], ascending=[True, False])
        clf_best = cands_sorted.drop_duplicates("gps_index").sort_values("gps_index")
        clf_pred_arcs = clf_best["arc_id"].astype(int).tolist()

        # 4. Road Graph Setup & Dijkstra Memoizer
        t_vit_start = time.time()
        G, arc_lookup, adj_weights = self.build_graph(arcs_df)
        sp_cache: Dict[Tuple[int, int], Dict[int, float]] = {}

        def get_sp_dict(source_node: int, horizon: float) -> Dict[int, float]:
            quantized_horizon = int(max(horizon, self.min_cutoff_m) / 25) * 25
            cache_key = (source_node, quantized_horizon)
            if cache_key not in sp_cache:
                sp_cache[cache_key] = nx.single_source_dijkstra_path_length(
                    G, source_node, cutoff=quantized_horizon, weight="weight"
                )
            return sp_cache[cache_key]

        # Group candidates by GPS observation index
        grouped = {}
        cand_records = df[[
            "gps_index", "arc_id", "clf_prob", "distance_to_arc_m", 
            "projection_fraction", "is_wrong_way"
        ]].to_dict("records")
        for rec in cand_records:
            grouped.setdefault(rec["gps_index"], []).append(rec)

        track_steps = {r.gps_index: r.step_distance_m for r in track_df.itertuples()}
        gps_indices = sorted(grouped.keys())

        if not gps_indices:
            return MatchResult(
                trip_id=trip_id,
                num_gps_points=0,
                classifier_top1_accuracy=None,
                viterbi_top1_accuracy=None,
                accuracy_gain=None,
                continuity_score_percent=0.0,
                classifier_arc_sequence=[],
                viterbi_arc_sequence=[],
                ground_truth_arc_sequence=None,
                feature_latency_ms=feat_lat_ms,
                inference_latency_ms=inf_lat_ms,
                viterbi_latency_ms=0.0,
                total_latency_ms=(time.time() - t_total_start) * 1000.0
            )

        # 5. Viterbi Initialization (t = 0)
        first_cands = grouped[gps_indices[0]]
        V_prev = np.zeros(len(first_cands))
        for i, c in enumerate(first_cands):
            p = max(c["clf_prob"], 1e-5)
            dist_term = -0.5 * ((c["distance_to_arc_m"] / self.sigma_z) ** 2)
            wrong_term = -self.wrong_way_penalty if c["is_wrong_way"] else 0.0
            V_prev[i] = np.log(p) + dist_term + wrong_term

        backpointers = []
        prev_cands = first_cands

        # 6. Viterbi Forward Recurrence (t = 1 .. T-1)
        for t in range(1, len(gps_indices)):
            curr_cands = grouped[gps_indices[t]]
            d_gps = track_steps.get(gps_indices[t], 10.0)
            cutoff = max(d_gps * 2.5 + 50.0, self.min_cutoff_m)

            V_curr = np.zeros(len(curr_cands))
            bp = np.zeros(len(curr_cands), dtype=int)

            n_prev = len(prev_cands)
            prev_arcs = [c["arc_id"] for c in prev_cands]
            prev_v = [arc_lookup[a][1] if a in arc_lookup else -1 for a in prev_arcs]
            prev_l = [arc_lookup[a][2] if a in arc_lookup else 50.0 for a in prev_arcs]
            prev_t = [c["projection_fraction"] for c in prev_cands]

            for j, c2 in enumerate(curr_cands):
                a2 = c2["arc_id"]
                u2, v2, l2 = arc_lookup.get(a2, (-1, -1, 50.0))
                t2 = c2["projection_fraction"]

                p2 = max(c2["clf_prob"], 1e-5)
                em_cost = (
                    np.log(p2)
                    - 0.5 * ((c2["distance_to_arc_m"] / self.sigma_z) ** 2)
                    - (self.wrong_way_penalty if c2["is_wrong_way"] else 0.0)
                )

                best_val = -1e9
                best_i = 0

                for i in range(n_prev):
                    a1 = prev_arcs[i]
                    if a1 == a2:
                        d_net = (t2 - prev_t[i]) * prev_l[i] if t2 >= prev_t[i] else (prev_t[i] - t2) * prev_l[i] + 5.0
                        trans_cost = -np.abs(d_net - d_gps) / self.beta
                    elif prev_v[i] == u2:
                        d_net = (1.0 - prev_t[i]) * prev_l[i] + t2 * l2
                        trans_cost = -np.abs(d_net - d_gps) / self.beta
                    elif prev_v[i] in adj_weights and u2 in adj_weights[prev_v[i]]:
                        d_net = (1.0 - prev_t[i]) * prev_l[i] + adj_weights[prev_v[i]][u2] + t2 * l2
                        trans_cost = -np.abs(d_net - d_gps) / self.beta
                    else:
                        sp = get_sp_dict(prev_v[i], cutoff)
                        if u2 in sp:
                            d_net = (1.0 - prev_t[i]) * prev_l[i] + sp[u2] + t2 * l2
                            trans_cost = -np.abs(d_net - d_gps) / self.beta
                        else:
                            trans_cost = -self.disconnected_penalty

                    candidate_total = V_prev[i] + trans_cost
                    if candidate_total > best_val:
                        best_val = candidate_total
                        best_i = i

                V_curr[j] = best_val + em_cost
                bp[j] = best_i

            backpointers.append(bp)
            V_prev = V_curr
            prev_cands = curr_cands

        # 7. Viterbi Backtracking
        best_last_idx = int(np.argmax(V_prev))
        path = [best_last_idx]
        for t in range(len(backpointers) - 1, -1, -1):
            best_last_idx = backpointers[t][best_last_idx]
            path.append(best_last_idx)
        path.reverse()

        viterbi_pred_arcs = [grouped[gps_indices[t]][path[t]]["arc_id"] for t in range(len(gps_indices))]
        vit_lat_ms = (time.time() - t_vit_start) * 1000.0

        # 8. Route Continuity Evaluation
        continuity = self.evaluate_continuity(viterbi_pred_arcs, G, arc_lookup)

        # 9. Ground Truth Accuracy Verification (Server-Side)
        clf_acc, vit_acc, gain, gt_arcs = None, None, None, None
        if "is_reference_candidate" in df.columns and df["is_reference_candidate"].any():
            ref_subset = df[df["is_reference_candidate"]].drop_duplicates("gps_index").sort_values("gps_index")
            gt_map = dict(zip(ref_subset["gps_index"], ref_subset["arc_id"].astype(int)))
            gt_arcs = [gt_map.get(idx, -1) for idx in gps_indices]

            hits_clf = sum(1 for p_arc, g_arc in zip(clf_pred_arcs, gt_arcs) if p_arc == g_arc and g_arc != -1)
            hits_vit = sum(1 for p_arc, g_arc in zip(viterbi_pred_arcs, gt_arcs) if p_arc == g_arc and g_arc != -1)
            n_eval = sum(1 for g_arc in gt_arcs if g_arc != -1)

            if n_eval > 0:
                clf_acc = (hits_clf / n_eval) * 100.0
                vit_acc = (hits_vit / n_eval) * 100.0
                gain = vit_acc - clf_acc

        total_lat_ms = (time.time() - t_total_start) * 1000.0

        return MatchResult(
            trip_id=trip_id,
            num_gps_points=len(gps_indices),
            classifier_top1_accuracy=clf_acc,
            viterbi_top1_accuracy=vit_acc,
            accuracy_gain=gain,
            continuity_score_percent=continuity,
            classifier_arc_sequence=clf_pred_arcs,
            viterbi_arc_sequence=viterbi_pred_arcs,
            ground_truth_arc_sequence=gt_arcs,
            feature_latency_ms=feat_lat_ms,
            inference_latency_ms=inf_lat_ms,
            viterbi_latency_ms=vit_lat_ms,
            total_latency_ms=total_lat_ms
        )
