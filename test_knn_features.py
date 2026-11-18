import pandas as pd
import numpy as np

base = "processed/train/00000034/"

c = pd.read_csv(base + "candidate_features.csv")
a = pd.read_csv(base + "arcs_features.csv")
t = pd.read_csv(base + "track_features.csv")

# Merge road network features
df = c.merge(
    a[["arc_id", "from_node", "to_node"]],
    on="arc_id",
    how="left"
)

# Merge GPS track features
df = df.merge(
    t[["gps_index", "speed_kmh", "heading_deg"]],
    on="gps_index",
    how="left"
)

# Basic derived features
df["log_distance"] = np.log1p(df["distance_to_arc_m"])
df["log_length"] = np.log1p(df["length_m"])
df["dist_to_length_ratio"] = (
    df["distance_to_arc_m"] / df["length_m"].replace(0, np.nan)
)

# Relative distance from the best candidate for each GPS point
best_distance = df.groupby("gps_index")["distance_to_arc_m"].transform("min")
df["relative_dist_to_best"] = df["distance_to_arc_m"] - best_distance

# Candidate density per GPS point
df["candidate_density"] = df.groupby("gps_index")["arc_id"].transform("count")

# Heading difference (circular difference)
diff = (df["bearing_deg"] - df["heading_deg"] + 180) % 360 - 180
df["heading_diff_deg"] = diff.abs()
df["cos_heading_diff"] = np.cos(np.radians(diff))

# Direction indicator (provisional; validate arc orientation before model use)
df["is_wrong_way"] = (np.abs(diff) > 90).astype(int)

# Road node degrees
from_degree = pd.concat([a["from_node"], a["to_node"]]).value_counts()
to_degree = from_degree

df["from_node_degree"] = df["from_node"].map(from_degree)
df["to_node_degree"] = df["to_node"].map(to_degree)

features = [
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

print("FEATURE ENGINEERING TEST")
print("=" * 50)

print("Total rows:", len(df))
print("Total features:", len(features))
print("\nFeature columns:")

for feature in features:
    print(feature, "->", "OK" if feature in df.columns else "MISSING")

print("\nMissing values:")
print(df[features].isnull().sum())

print("\nFeature preview:")
print(df[features].head())

df[features + ["is_reference_candidate"]].to_csv(
    "knn_feature_test.csv",
    index=False
)

print("\nSaved: knn_feature_test.csv")
