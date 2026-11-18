# PathSnap: Distributed Map-Matching & Route Reconstruction

## 1. Project Overview

**PathSnap** is a Machine Learning-based map-matching and route reconstruction project that aims to identify the actual road segments travelled by a vehicle using noisy GPS trajectory data and a digital road network.

GPS devices frequently produce inaccurate location readings due to signal errors, environmental interference and nearby parallel roads. Consequently, the recorded GPS coordinates may not accurately represent the roads on which a vehicle has travelled.

PathSnap addresses this problem by processing GPS trajectories, generating possible road-segment candidates, applying Machine Learning models to identify the most likely candidates, and using probabilistic sequence modelling to reconstruct a consistent vehicle route.

The project combines data processing, feature engineering, supervised Machine Learning, Hidden Markov Models (HMM) and the Viterbi algorithm.

## 2. Team Details

**Team Number: 2**

| Sl. No. | Student Name | SRN |
|---|---|---|
| 1 | Neel Chandrakar | PES2UG24CS310 |
| 2 | Neeraj R Rugi | PES2UG24CS311 |
| 3 | Prarthana Herbur | PES2UG24CS367 |
| 4 | Pratham Kumar | PES2UG24CS368 |

## 3. Problem Statement

GPS trajectories contain positional inaccuracies that make it difficult to determine the exact road segments travelled by a vehicle.

A GPS coordinate may be closer to an incorrect neighbouring road than to the actual road. Selecting the nearest road independently for every GPS observation can also produce an unrealistic or disconnected route.

Therefore, the objective of PathSnap is to develop an intelligent map-matching framework that:

- Processes raw GPS trajectory data.
- Represents the available road network.
- Generates candidate road segments for each GPS observation.
- Extracts meaningful geometric and movement-related features.
- Identifies the most likely road candidates using Machine Learning.
- Uses sequential probability modelling to maintain route continuity.
- Reconstructs the most likely sequence of road segments.

## 4. Project Workflow

The overall project follows these stages:

1. Dataset Understanding and Exploration
2. GPS Data Processing and Preprocessing
3. Road Network Processing
4. Candidate Road Segment Generation
5. Feature Engineering
6. Reference-Based Label Generation
7. Machine Learning Model Training and Comparison
8. Selection of the Best-Performing Model
9. Hidden Markov Model Integration
10. Viterbi-Based Sequence Decoding
11. Final Route Reconstruction and Evaluation

## 5. Dataset Description

The project uses GPS trajectory data along with corresponding road-network information.

The dataset contains the following file types:

| File | Description |
|---|---|
| `.track` | Contains GPS longitude, latitude and timestamp observations. |
| `.nodes` | Contains road-network node information and coordinates. |
| `.arcs` | Represents directed connections between road-network nodes. |
| `.route` | Contains reference route information used for labelling and evaluation. |

### Dataset Split

The project dataset follows the provided 80/10/10 distribution:

- **80% – Training data:** Used for model training.
- **10% – Student testing data:** Used for model evaluation and development.
- **10% – Faculty-side testing data:** Reserved for independent evaluation.

The faculty-side testing data is maintained separately and should not be used during model training or hyperparameter selection.

## 6. Data Processing and Feature Engineering

The initial phase focuses on preparing the GPS observations and road-network information for Machine Learning.

Major operations include:

- Loading and examining GPS trajectory files.
- Processing road-network nodes and directed arcs.
- Cleaning and preparing GPS observations.
- Generating possible road-segment candidates near GPS coordinates.
- Calculating geometric relationships between GPS observations and candidate roads.
- Extracting road and movement-related features.
- Generating reference-based candidate labels.

The engineered features provide information about the relationship between GPS observations and possible road segments.

Examples include:

- Distance between GPS observations and candidate road segments.
- Road-segment length.
- Road bearing.
- Projection fraction.
- GPS movement-related features, where available.
- Other derived geometric and spatial features.

The processed and labelled candidate dataset is subsequently used for supervised Machine Learning.

## 7. Machine Learning Model Experimentation

Five supervised Machine Learning algorithms are considered and evaluated to identify the most suitable model for road-candidate classification.

The algorithms include:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Logistic Regression

The models are trained using the engineered candidate features and reference-based labels.

Their performance is compared using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The comparison helps identify the algorithm that provides the most suitable candidate-classification performance.

## 8. Selected Model: XGBoost

Based on the initial comparative evaluation, **XGBoost (Extreme Gradient Boosting)** was selected as the leading model, achieving approximately 85% reported accuracy in the team's experiments.

XGBoost is a gradient-boosting algorithm that builds decision trees sequentially, with successive trees improving the errors made by previous trees.

In PathSnap, XGBoost is used to:

- Learn patterns from engineered road-candidate features.
- Classify candidate road segments.
- Estimate the likelihood of each candidate being correct.
- Generate candidate scores that can be used by the subsequent probabilistic sequence-modelling stage.

Further evaluation and optimisation are intended to improve the model's performance and generalisation.

*Note: The approximately 85% accuracy is an initial reported experimental result. Final performance should be documented with the corresponding test split, classification metrics and GPS observation-level evaluation results.*

## 9. Hidden Markov Model (HMM)

After selecting the Machine Learning model, the next stage focuses on sequential route reconstruction using a Hidden Markov Model.

In the map-matching problem:

- **Hidden states:** Actual road segments travelled by the vehicle.
- **Observations:** Recorded GPS coordinates.
- **Emission probabilities:** Represent how likely a GPS observation is associated with a candidate road.
- **Transition probabilities:** Represent how likely movement between consecutive road candidates is based on road-network connectivity and movement information.

The selected XGBoost model provides candidate-likelihood scores that can be incorporated into the emission-probability calculation.

The transition model considers the road network and the feasibility of movement between candidate road segments.

## 10. Viterbi Algorithm

The Viterbi algorithm is used to identify the most likely sequence of hidden states in the Hidden Markov Model.

Selecting the highest-scoring road candidate independently for every GPS observation does not necessarily produce a continuous or realistic route.

Viterbi addresses this by considering:

- Candidate emission probabilities.
- Road-to-road transition probabilities.
- The likelihood of the complete candidate sequence.

It finds the most likely sequence of road segments across the GPS trajectory.

The resulting sequence is used for final route reconstruction.

## 11. Overall System Architecture

```text
              GPS Trajectory Dataset
                       |
                       v
              Road Network Dataset
                       |
                       v
          Data Processing & Preprocessing
                       |
                       v
             Candidate Generation
                       |
                       v
              Feature Engineering
                       |
                       v
          Reference-Based Label Generation
                       |
                       v
          Machine Learning Experimentation
                       |
        +--------------+--------------+
        |              |              |
   Decision Tree   Random Forest     SVM
        |              |              |
       KNN       Logistic Regression  |
        |              |              |
        +--------------+--------------+
                       |
                       v
             Model Performance Comparison
                       |
                       v
                 XGBoost Selection
                       |
                       v
             Candidate Probability Scores
                       |
                       v
          Hidden Markov Model (HMM)
                       |
                       v
              Viterbi Algorithm
                       |
                       v
            Final Route Reconstruction
                       |
                       v
              Route Evaluation
```

## 12. Evaluation Strategy

The project evaluates the Machine Learning models and final route reconstruction using appropriate performance measures.

### Classification Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

### Map-Matching Evaluation

- GPS observation-level Top-1 accuracy.
- Correct candidate selection rate.
- Route reconstruction performance.
- Route continuity and transition validity.
- Comparison against reference routes.

Classification performance and final route-reconstruction performance are evaluated separately because a correct candidate-level prediction does not automatically guarantee a completely correct route sequence.

## 13. Expected Outcome

The expected outcome of PathSnap is a map-matching framework capable of transforming noisy GPS trajectory observations into a logically consistent sequence of road segments.

By combining feature engineering, supervised Machine Learning, probabilistic modelling and sequence decoding, the project aims to improve road-candidate identification and vehicle route reconstruction.

## 14. Technologies and Libraries

The implementation uses Python and relevant Machine Learning and geospatial libraries, including:

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- GeoPandas / Shapely, where applicable
- NetworkX, where applicable
- Jupyter Notebook / Google Colab

Additional libraries may be used for geospatial processing, route visualisation and model evaluation.

## 15. Repository Structure

The repository is organised around the project's data-processing, feature-engineering, Machine Learning and route-reconstruction stages.

```text
ML_Hackathon/
│
├── Dataset/
│   ├── train/
│   └── student_test/
│
├── data_processing/
│
├── feature_engineering/
│
├── model_training/
│   ├── decision_tree/
│   ├── random_forest/
│   ├── svm/
│   ├── knn/
│   ├── logistic_regression/
│   └── xgboost/
│
├── hmm/
│
├── viterbi/
│
├── route_reconstruction/
│
├── evaluation/
│
├── visualizations/
│
└── README.md
```

*The structure above describes the logical organisation of the project. Actual directory names should be kept consistent with the repository implementation.*

## 16. Conclusion

PathSnap explores an integrated Machine Learning and probabilistic approach to GPS map matching and route reconstruction.

The project begins with GPS and road-network data processing, followed by candidate generation and feature engineering. Five supervised Machine Learning algorithms are evaluated, with XGBoost emerging as the leading model in the initial experiments.

The selected model's candidate-likelihood scores are intended to support an HMM-based sequential modelling approach, while the Viterbi algorithm identifies the most likely sequence of road segments.

The overall objective is to reconstruct realistic and consistent vehicle routes from noisy GPS trajectory data.
