# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 21:04:32 2026

@author: Margaux Thieury
"""
"""
Display results of SVM classification

"""
#%%
"""
Figure 3
Plot SVM classification results for results on 2 conditions

This script:
1) Loads prediction results from multiple experiments
2) Computes performance metrics (accuracy, F1, mutual info)
3) Creates a boxplot with overlayed points per frequency band
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mutual_info_score, adjusted_mutual_info_score

# =============================================================================
# Paths to experiment result folders
# =============================================================================
base_paths = {
    "J_LM": '.../results_prediction/Diel_HabitatMonth/',
    "L_MJ": '.../results_prediction/Habitat_MonthDiel/',
    "M_LJ": '.../results_prediction/Month_HabitatDiel/'
}

# =============================================================================
# Load all CSV results and compute metrics
# =============================================================================
all_results = []

for exp_name, path in base_paths.items():
    # List all subfolders (one per condition)
    subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for folder in subfolders:
        folder_path = os.path.join(path, folder)
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        for file in files:
            df = pd.read_csv(os.path.join(folder_path, file))

            # Check if required columns exist
            if {"true_label_name", "predicted_label_name"}.issubset(df.columns):
                y_true = df["true_label_name"]
                y_pred = df["predicted_label_name"]
                labels_true = df["true_label"]
                labels_pred = df["predicted_label"]

                # Compute mutual information
                MI = mutual_info_score(labels_true, labels_pred)
                AMI = adjusted_mutual_info_score(labels_true, labels_pred)

                # Classification report for accuracy and F1
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                accuracy = report.get("accuracy", None)
                f1_macro = report.get("macro avg", {}).get("f1-score", None)

                # Identify band from filename
                band = next((b for b in ["audible", "ultrasonic", "large"] if b in file), "unknown")

                # Store class-wise F1-scores
                class_f1_scores = {label: report[label]["f1-score"] 
                                   for label in report if label not in ["accuracy", "macro avg", "weighted avg"]}

                # Append results
                all_results.append({
                    "Folder": folder,
                    "File": file,
                    "Band": band,
                    "Accuracy": accuracy,
                    "F1-score_G": f1_macro,
                    "Mutual_info": MI,
                    "Adjusted_Mutual_info": AMI,
                    "Experiment": exp_name,
                    **class_f1_scores
                })

# Convert to DataFrame
df_all = pd.DataFrame(all_results)

# =============================================================================
# Plotting
# =============================================================================
sns.set_theme(style="ticks")

selected_metric = "Accuracy"  # Choose metric to display: "Accuracy", "F1-score_G", "Adjusted_Mutual_info", etc.

# Initialize figure
fig, ax = plt.subplots(figsize=(10, 5))

# Custom color palette for bands
palette = {
    "audible": "#7f56c8ff",     # blue
    "ultrasonic": "#e95e50ff",  # orange
    "large": "#ffc822ff"        # green
}

# Boxplot per experiment and frequency band
sns.boxplot(
    data=df_all,
    x="Experiment",
    y=selected_metric,
    hue="Band",
    whis=[0, 100],              # min-max whiskers
    width=0.4,
    palette=palette,
    order=["J_LM", "L_MJ", "M_LJ"],
    hue_order=["audible", "ultrasonic", "large"]
)

# Overlay individual points (stripplot)
sns.stripplot(
    data=df_all,
    x="Experiment",
    y=selected_metric,
    hue="Band",
    dodge=True,
    jitter=0.03,
    size=3,
    color=".3",
    alpha=0.7,
    legend=False,  # avoid double legend
    order=["J_LM", "L_MJ", "M_LJ"],
    hue_order=["audible", "ultrasonic", "large"]
)

# Visual adjustments
ax.yaxis.grid(True)
plt.ylim(0, 1)
ax.set(xlabel="Experiment", ylabel=selected_metric)

# Increase font sizes
plt.xlabel("Experiment", fontsize=14)
plt.ylabel(selected_metric, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Legend customization
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], ['audible (150-18kHz)', 'ultrasonic (18-36kHz)', 'all (150-36kHz)'], 
          title="Frequencies", loc="lower right", fontsize=14)

sns.despine(trim=True, left=True)
plt.tight_layout()
plt.show()

#%%

"""
Figure 4
Interactive F1-score plotting for SVM classification results

User selects:
1) Main experiment folder
        MonthHabitatDiel
    1-condition:
        Diel_HabitatMonth
        Habitat_MonthDiel
        Month_HabitatDiel
    2-conditions:
        HabitatDiel_Month
        HabitatMonth_Diel
        MonthDiel_Habitat
2) Subfolder (condition) inside the experiment
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mutual_info_score, adjusted_mutual_info_score

# =============================================================================
# Base path containing all experiments
# =============================================================================
BASE_PATH = r"...\results_prediction"

# List all experiments
experiments = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
print("Available experiments:")
for i, exp in enumerate(experiments):
    print(f"{i}: {exp}")

# User chooses experiment
exp_idx = int(input("Enter the number of the experiment to analyze: "))
selected_exp = experiments[exp_idx]
exp_path = os.path.join(BASE_PATH, selected_exp)
print(f"Selected experiment: {selected_exp}")

# =============================================================================
# List all subfolders inside the selected experiment
# =============================================================================
subfolders = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]

print("\nAvailable subfolders in the selected experiment:")
for i, folder in enumerate(subfolders):
    print(f"{i}: {folder}")

# User chooses subfolder
folder_idx = int(input("Enter the number of the folder to analyze: "))
selected_folder = subfolders[folder_idx]
folder_path = os.path.join(exp_path, selected_folder)
print(f"Selected folder: {selected_folder}")

# =============================================================================
# Load all CSV results from the chosen subfolder
# =============================================================================
accuracy_results = []

for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)

        if {"true_label_name", "predicted_label_name"}.issubset(df.columns):
            y_true = df["true_label_name"]
            y_pred = df["predicted_label_name"]
            labels_true = df["true_label"]
            labels_pred = df["predicted_label"]

            MI = mutual_info_score(labels_true, labels_pred)
            AMI = adjusted_mutual_info_score(labels_true, labels_pred)

            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            accuracy = report.get("accuracy", None)
            f1_macro = report.get("macro avg", {}).get("f1-score", None)

            band = next((b for b in ["audible", "ultrasonic", "large"] if b in file), "unknown")
            class_f1_scores = {label: report[label]["f1-score"] 
                               for label in report if label not in ["accuracy", "macro avg", "weighted avg"]}

            accuracy_results.append({
                "Folder": selected_folder,
                "Mutual_info": MI,
                "Adjusted_Mutual_info": AMI,
                "File": file,
                "Band": band,
                "Accuracy": accuracy,
                "F1-score_G": f1_macro,
                **class_f1_scores
            })

df_accuracy = pd.DataFrame(accuracy_results)

# =============================================================================
# Prepare data for plotting
# =============================================================================
df_melted = df_accuracy.melt(
    id_vars=["Folder", "Band", "File"],
    value_vars=[col for col in df_accuracy.columns if col not in ["Folder", "File", "Accuracy", "Band"]],
    var_name="Classe",
    value_name="F1-score"
)

# Optional: rename classes
rename_classes = {
    "DECEMBRE2022": "December",
    "JULY2023": "July",
    "MARCH2023": "March",
    "MAY2022": "May",
    "OCTOBER2023": "October"
}
df_melted["Classe"] = df_melted["Classe"].replace(rename_classes)

# Filter out global metrics
df_filtered = df_melted[~df_melted["Classe"].isin(["F1-score_G", "Mutual_info", "Adjusted_Mutual_info"])]

# =============================================================================
# Plotting function
# =============================================================================
def plot_f2_boxplot(df, folder_name):
    """
    Plot F1-score boxplot for only 'audible' and 'large' bands with custom colors.

    Parameters:
        df (DataFrame): Filtered dataframe with columns ["Classe", "F1-score", "Band"]
        folder_name (str): Name of the folder/category for labeling

    Returns:
        fig: Matplotlib figure object
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Only select audible and large bands
    df_filtered = df[df["Band"].isin(["audible", "large"])]

    # Frequency bands and custom colors
    band_order = ["audible", "large"]
    palette = {
        "audible": "#7f56c8ff",   # blue
        "large": "#ffc822ff"      # yellow/orange
    }

    # Determine the classes dynamically from the dataframe
    classe_order = df_filtered["Classe"].unique().tolist()
    if not classe_order:
        print("No classes found in the selected folder. Cannot plot.")
        return fig

    # Boxplot
    sns.boxplot(
        data=df_filtered,
        x="Classe",
        y="F1-score",
        hue="Band",
        whis=[0, 100],
        width=0.6,
        palette=palette,
        order=classe_order,
        hue_order=band_order,
        ax=ax
    )

    # Stripplot overlay
    sns.stripplot(
        data=df_filtered,
        x="Classe",
        y="F1-score",
        hue="Band",
        dodge=True,
        jitter=0.03,
        size=3,
        color=".3",
        alpha=0.7,
        legend=False,
        order=classe_order,
        hue_order=band_order,
        ax=ax
    )

    # Axis formatting
    ax.set_xlabel("Categories", fontsize=16)
    ax.set_ylabel("F1-score", fontsize=16)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True)
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(fontsize=14)

    # Remove duplicate legend
    if ax.get_legend() is not None:
        ax.legend_.remove()

    sns.despine(trim=True, left=True)
    plt.tight_layout()
    plt.show()

    return fig

# =============================================================================
# Plot
# =============================================================================
plot_f2_boxplot(df_filtered, selected_folder)

#%% Results for Figure 5. Quantifying the informational contribution of ultrasound

# =============================================================================
# Compute ΔAMI = AMI(large) - AMI(ultrasonic) for each experiment/folder
# =============================================================================

import os
import pandas as pd
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score


# -----------------------------------------------------------------------------
# Paths for each experiment
# -----------------------------------------------------------------------------
base_chemins = {
    "Diel_HabitatMonth": '.../results_prediction/Diel_HabitatMonth/',
    "Habitat_MonthDiel": '.../results_prediction/Habitat_MonthDiel/',
    "Month_HabitatDiel": '.../results_prediction/Month_HabitatDiel/',
    "HabitatDiel_Month": '.../results_prediction/HabitatDiel_Month/',
    "HabitatMonth_Diel": '.../results_prediction/HabitatMonth_Diel/',
    "MonthDiel_Habitat": '.../results_prediction/MonthDiel_Habitat/',
    "MonthHabitatDiel":  '.../results_prediction/MonthHabitatDiel/'
}


# -----------------------------------------------------------------------------
# Collect AMI per file
# -----------------------------------------------------------------------------
records = []

for exp_name, root in base_chemins.items():

    folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    for folder in folders:

        folder_path = os.path.join(root, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

        for file in files:

            df = pd.read_csv(os.path.join(folder_path, file))

            if not {"true_label", "predicted_label"}.issubset(df.columns):
                continue

            y_true = df["true_label"]
            y_pred = df["predicted_label"]

            ami = adjusted_mutual_info_score(y_true, y_pred)

            band = next((b for b in ["audible", "ultrasonic", "large"] if b in file), "unknown")

            records.append({
                "Experiment": exp_name,
                "Folder": folder,
                "Band": band,
                "AMI": ami
            })


df_all = pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Average AMI per (Experiment, Folder, Band)
# -----------------------------------------------------------------------------
df_mean = (
    df_all
    .groupby(["Experiment", "Folder", "Band"], as_index=False)["AMI"]
    .mean()
)


# -----------------------------------------------------------------------------
# Pivot → compute ΔAMI
# -----------------------------------------------------------------------------
df_pivot = df_mean.pivot_table(
    index=["Experiment", "Folder"],
    columns="Band",
    values="AMI"
).reset_index()

df_pivot["Delta_AMI"] = df_pivot["large"] - df_pivot["audible"]
df_pivot = df_pivot.round(3)


# -----------------------------------------------------------------------------
# Display results
# -----------------------------------------------------------------------------
for exp_name in base_chemins.keys():

    print(f"\n=== {exp_name} | ΔAMI (large - ultrasonic) ===")

    df_exp = df_pivot[df_pivot["Experiment"] == exp_name]

    print(df_exp[["Folder", "large", "ultrasonic", "Delta_AMI"]])
