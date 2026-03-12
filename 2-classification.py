# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 19:23:50 2026

@author: Margaux Thieury
"""
# -*- coding: utf-8 -*-
"""
SVM classification pipeline for FFT features

Steps:
1) Load all CSV and concatenate
2) Global MLJ experiment (Month + Location + DayNight)
3) Single-condition experiments
4) Two-condition experiments
"""

# =============================================================================
# Load all CSVs
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


FOLDER = r'features_256'
csv_files = glob.glob(os.path.join(FOLDER, '*.csv'))

df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print("Data loaded:", df_all.shape)


# =============================================================================
# Common parameters
# =============================================================================

BASE_DIR = r"results_prediction/"
os.makedirs(BASE_DIR, exist_ok=True)

N_bins = 256

train_test_splits = [
    ([0, 1], 2),
    ([1, 2], 0),
    ([0, 2], 1)
]


# =============================================================================
# Utility: extract FFT band features
# =============================================================================

def extract_band_features(df, band, target):

    bins = np.arange(N_bins)
    freqs = bins * (48000 / N_bins)

    ranges = {
        'audible': (150, 18000),
        'ultrasonic': (18000, 35850),
        'large': (150, 35850)
    }

    fmin, fmax = ranges[band]
    mask = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    cols = [f"bin_reduced_{b}" for b in mask]

    return df[cols].values, df[target].values


# =============================================================================
# Generic SVM runner
# =============================================================================

def run_svm(df, target_col, out_dir, encoder):

    for band in ['audible', 'ultrasonic', 'large']:

        for train_days, test_day in train_test_splits:

            df_train = df[df['label_DAY_encoded'].isin(train_days)]
            df_test  = df[df['label_DAY_encoded'] == test_day]

            if df_train.empty or df_test.empty:
                continue

            X_train, y_train = extract_band_features(df_train, band, target_col)
            X_test, y_test   = extract_band_features(df_test, band, target_col)

            clf = SVC(
                C=1,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=24
            )

            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_test)
            preds = clf.classes_[np.argmax(probs, axis=1)]

            df_res = df_test[['path']].copy()

            for i, c in enumerate(clf.classes_):
                df_res[f"prob_class_{c}"] = probs[:, i]

            df_res["predicted_label"] = preds
            df_res["predicted_label_name"] = encoder.inverse_transform(preds)
            df_res["true_label"] = y_test
            df_res["true_label_name"] = encoder.inverse_transform(y_test)

            out = os.path.join(out_dir, f"results_{band}_train_{train_days}_test_{test_day}.csv")
            df_res.to_csv(out, index=False)


# =============================================================================
# 1) MLJ experiment (Month + Location + DayNight)
# =============================================================================

df = df_all.copy()

df["MLJ"] = df["label_MONTH"] + "_" + df["label_LOCATION"] + "_" + df["label_DAYNIGHT"]

enc = LabelEncoder()
df["MLJ_encoded"] = enc.fit_transform(df["MLJ"])

run_svm(df, "MLJ_encoded", os.path.join(BASE_DIR, "MonthHabitatDiel"), enc)

print("MLJ done.")


# =============================================================================
# 2) Single-condition experiments
# =============================================================================

def process_single(cond, new_name, folder, c1, c2):

    for val in df_all[cond].unique():

        d = df_all[df_all[cond] == val].copy()

        for col in ["label_MONTH", "label_LOCATION", "label_DAYNIGHT"]:
            d[f"{col}_encoded"] = LabelEncoder().fit_transform(d[col])

        d[new_name] = d[c1] + "_" + d[c2]

        enc = LabelEncoder()
        d[f"{new_name}_encoded"] = enc.fit_transform(d[new_name])

        out_dir = os.path.join(BASE_DIR, folder, str(val))
        os.makedirs(out_dir, exist_ok=True)

        run_svm(d, f"{new_name}_encoded", out_dir, enc)


process_single("label_MONTH",     "LJ", "HbaitatDiel_Month", "label_LOCATION", "label_DAYNIGHT")
process_single("label_DAYNIGHT",  "LM", "HabitatMonth_Diel", "label_LOCATION", "label_MONTH")
process_single("label_LOCATION",  "MJ", "MonthDiel_Habitat", "label_MONTH",    "label_DAYNIGHT")

print("Single-condition done.")


# =============================================================================
# 3) Two-condition experiments
# =============================================================================

def process_double(c1, c2, folder, target):

    for v1 in df_all[c1].unique():
        for v2 in df_all[c2].unique():

            d = df_all[(df_all[c1] == v1) & (df_all[c2] == v2)].copy()
            if d.empty:
                continue

            for col in ["label_MONTH", "label_LOCATION", "label_DAYNIGHT"]:
                d[f"{col}_encoded"] = LabelEncoder().fit_transform(d[col])

            enc = LabelEncoder()
            d[f"{target}_encoded"] = enc.fit_transform(d[target])

            out_dir = os.path.join(BASE_DIR, folder, f"{v1}_{v2}")
            os.makedirs(out_dir, exist_ok=True)

            run_svm(d, f"{target}_encoded", out_dir, enc)


process_double("label_MONTH",    "label_DAYNIGHT", "Habitat_MonthDiel", "label_LOCATION")
process_double("label_LOCATION", "label_DAYNIGHT", "Month_HabitatDiel", "label_MONTH")
process_double("label_LOCATION", "label_MONTH",    "Diel_HabitatMonth", "label_DAYNIGHT")

print("\nAll experiments completed.")

