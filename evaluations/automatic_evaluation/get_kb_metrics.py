import pandas as pd

# Load dataset (use your aggregated_scores_data.csv file)
file_path = "evaluations/automatic_evaluation/aggregated_scores_data.csv"
data = pd.read_csv(file_path)


# KB mapping similar to create_scatterplot.py
def map_kb(folder_name):
    if "RelatedTo" in folder_name or "UsedFor" in folder_name:
        return "ConceptNet"
    elif "gloss" in folder_name or "nogloss" in folder_name:
        return "FrameNet"
    elif "multialignet" in folder_name:
        return "MultiAlignet"
    elif "semagram" in folder_name:
        return "Semagram"
    return folder_name


# Apply KB and extract model from "Parent Folder"
data["KB"] = data["Parent Folder"].apply(map_kb)
data["Model"] = data["Parent Folder"].apply(
    lambda x: x.split("/")[1] if "/" in x else x
)

# Temp fix: for Jamba-v0.1 filter out problematic rows where R@1 > 1.0
data.loc[(data["Model"] == "Jamba-v0.1") & (data["R@1"] > 1.0), "R@1"] = None

# Filter to one-shot experiments (Filename contains "oneshot")
data_oneshot = data[data["Filename"].str.contains("oneshot", case=False, na=False)]

# Group by KB and compute mean P@10, R@10, and MRR
agg = data_oneshot.groupby("KB")[["P@10", "R@10", "MRR"]].mean()

# Compute F1@10 using F1 = (2 * P@10 * R@10) / (P@10 + R@10); avoid division by zero.
agg["F1@10"] = agg.apply(
    lambda row: (2 * row["P@10"] * row["R@10"] / (row["P@10"] + row["R@10"]))
    if (row["P@10"] + row["R@10"]) > 0
    else 0,
    axis=1,
)

# Show the consolidated metrics
print("KB Metrics (One-Shot Only):")
print(agg[["P@10", "R@10", "F1@10", "MRR"]])
