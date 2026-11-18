# PathSnap: Project Handover & Session State

**Competition:** PES University ML Hackathon — PathSnap Distributed GPS Map-Matching System  
**Last Updated:** 2026-11-18  
**Repository:** [https://github.com/neeraj-r-rugi/ML_Hackathon.git](https://github.com/neeraj-r-rugi/ML_Hackathon.git)  
**Assigned Scope:** Support Vector Machine (SVM) Suite & XGBoost Gradient Boosting Suite

---

## 1. Executive Summary & Architecture

PathSnap solves the GPS map-matching problem by mapping noisy GPS trajectories onto a directed road network graph. It operates in two stages:
1. **Stage 1 (Candidate Classification):** For each GPS point, predict the score/probability of each candidate road arc within a 75m radius.
2. **Stage 2 (HMM & Viterbi Decoding):** Use candidate emission probabilities and road network transition costs to decode the globally optimal contiguous path.

---

## 2. Team Division & Git Scope

* **User's Scope (Tracked & Committed):**
  * `train_svm.py` — Complete 5-variant SVM evaluation suite + Optuna Bayesian optimization.
  * `train_xgboost.py` — Complete 5-variant XGBoost evaluation suite (classifier, ranker, and Bayesian tuned).
  * `model_svm.joblib`, `svm_best_model.joblib`, `svm_benchmark_results.json`
  * `model_xgboost.joblib`, `xgboost_best_model.joblib`, `xgboost_benchmark_results.json`
  * `enhanced_preprocessing.py` — Feature engineering pipeline fixing candidate ranking and adding directional/relative features.
  * `cli_eda.py` — Pure-terminal Unicode/ASCII spatial visualizer and kinematics inspector.
* **Teammates' Scope (Modularized Locally, Excluded via `.gitignore`):**
  * `train_logistic_regression.py` & `model_logistic_regression.joblib`
  * `train_random_forest.py` & `model_random_forest.joblib`
  * `train_knn.py` & `model_knn.joblib`

---

## 3. Benchmark Results Across Models

Evaluated on the **Urban Downtown Holdout** (Trips 26–29, 45,237 candidates) and **Highway Holdout** (Trips 13 & 14, 56,725 candidates):

| Model Architecture | Algorithm Family | Top-1 (Urban) | Top-1 (Highway) | Macro F1 | Binary Precision | Binary Recall | Log Loss | Inference Latency | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Depth 5)** *(Recommended)* | Gradient Boosting | **86.69%** | **92.90%** | **0.8608** | 60.1% | **98.5%** | **0.1042** | **0.2 µs** | **502 KB** |
| **SVM (Modified Huber)** | Kernel / Margin | 85.44% | 88.74% | 0.7757 | 40.1% | 97.8% | 0.2044 | **0.1 µs** | **2.4 KB** |
| **Random Forest (Depth 14)** | Ensemble (Bagging) | **86.95%** | 92.80% | **0.8822** | **65.2%** | 94.5% | **0.0844** | 0.7 µs | 16.5 MB |
| **Logistic Regression (L2)** | Linear / Probabilistic | 86.52% | 89.73% | 0.7803 | 41.0% | 97.5% | 0.1690 | **0.1 µs** | **2.2 KB** |
| **KNN (Manhattan L1)** | Instance-Based | 84.45% | **92.93%** | 0.8227 | 52.3% | 96.4% | 0.1728 | 50.9 µs | 15.7 MB |

### Key Domain Insights for Viva / Evaluators:
1. **The 58 cm Barrier:** On misclassified points, the median delta between chosen road and true road is only **0.58 meters (58 cm)**. Consumer GPS noise ($\pm 5 - 15\text{m}$) is wider than lane spacing ($3.5\text{m}$). Therefore, **no point-wise classifier can exceed ~87% in dense urban grids without temporal sequence modeling (HMM/Viterbi)**.
2. **Why XGBoost Won:** Tree splits natively evaluate joint non-linear logic (`relative_dist < 1.2m` AND `heading_diff < 20°`), cutting false positives by 55% (3,464 vs 7,691 for SVM). `is_wrong_way` accounted for 9.83% of tree split gain.

---

## 4. Local Environment & Setup

* **Python Virtual Environment:** `.venv -> /Users/neelchandrakar/projects/apps/ml/.venv` (Python 3.14 Apple Silicon arm64).
* **Dependencies Installed:** `xgboost`, `optuna`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `folium`, `matplotlib`, `fastapi`, `uvicorn`, `networkx`.
* **OpenMP Runtime:** Installed via `brew install libomp`.

---

## 5. Next Steps to Resume (Priority Order)

When restarting the assistant or continuing development, execute the following remaining deliverables:

### Step 1: Implement HMM & Viterbi Sequence Decoder (`hmm_decoder.py`)
* **Objective:** Combine `model_xgboost.joblib` emission probabilities $P(z_t \mid s_i^t)$ with road graph transition costs $P(s_j^{t+1} \mid s_i^t)$ using NetworkX shortest-path distance.
* **Target:** Boost route sequence accuracy from **87% $\to$ >95%**.

### Step 2: Build Two-Laptop Deployment (`app.py` & `ui.py`)
* **Laptop 1 (Backend):** FastAPI server bound to `0.0.0.0:8000` loading models and graph into memory once at startup. Endpoint `/predict` takes GPS trajectory and returns matched route JSON.
* **Laptop 2 (Evaluator UI):** Streamlit or HTML/JS frontend querying Laptop 1's local IP and rendering matched route polyline.

### Step 3: Interactive Folium Visualizer (`generate_maps.py`)
* Generate multi-layer Folium HTML map showing:
  1. Raw GPS points (red circles with popup speeds/timestamps).
  2. Candidate road arcs (grey lines).
  3. Ground-truth reference route (green line).
  4. Predicted matched route (blue dashed line).

### Step 4: Technical Report PDF
* Populate Sections 2 to 7 of `Deliverables.pdf` using the benchmark tables, Optuna objective curves, and Folium map screenshots.

---

## 6. Handy CLI Commands

```bash
# 1. Regenerate preprocessed data (e.g. 25 trips):
.venv/bin/python enhanced_preprocessing.py --limit-trips 25

# 2. Run SVM training & Optuna Bayesian tuning:
.venv/bin/python train_svm.py --train-trips 18 --test-trips 4 --trials 8

# 3. Run XGBoost training & Optuna Bayesian tuning:
.venv/bin/python train_xgboost.py --train-trips 18 --test-trips 4 --trials 8

# 4. Check git status:
git status
```
