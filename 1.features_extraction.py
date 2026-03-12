# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 18:38:42 2026

@author: Margaux Thieury
"""

"""
Acoustic Feature Extraction for Mozambique Dataset
-------------------------------------------------

This script extracts spectral acoustic features from 1-minute WAV segments.

Processing pipeline for each file:
    1. STFT using 100 ms Hann windows with 50% overlap
    2. Magnitude spectrum
    3. Temporal averaging across all frames
    4. Frequency averaging (dimension reduction → 256 bins)
    5. Log scaling

Output:
    One CSV per logger containing:
        - metadata (date, time, month, day/night, location)
        - 256 spectral features (freq_bin_0 ... freq_bin_255)

"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
from datetime import datetime, timedelta

import ephem
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# CONFIGURATION
# ============================================================================

# Geographic coordinates (Maputo)
LATITUDE = -26.5480
LONGITUDE = 32.7761

# -----------------------
# Audio parameters
# -----------------------
SR = 96000                 # sampling rate (Hz)
WINDOW_LENGTH = 0.1        # 100 ms
HOP_LENGTH = 0.05          # 50% overlap
N_FFT = 9600               # gives 10 Hz resolution

N_BINS_RAW = 4801          # raw STFT bins (n_fft/2 + 1)
N_BINS = 256               # final reduced bins after averaging


# -----------------------
# Campaign paths
# -----------------------
# Campaign definitions
CAMPAIGN_PATHS = {
    # 'C1_May2022': '.../C1_MSR_May2022/',
    # 'C2_December2022': '.../C2_MSR_December2022/',
    # 'C3_March2023': '.../C3_MSR_March2023/',
    # 'C4_July2023': '.../C4_MSR_July2023/',
    'C5_October2023': '.../C5_MSR_October2023/'
}

# Selected dates (3 days per campaign)
SELECTED_DATES = {
    'C1_May2022': ["2022-05-18", "2022-05-22", "2022-05-24"],
    'C2_December2022': ["2022-12-04", "2022-12-07", "2022-12-10"],
    'C3_March2023': ["2023-03-11", "2023-03-17", "2023-03-22"],
    'C4_July2023': ["2023-07-01", "2023-07-05", "2023-07-13"],
    'C5_October2023': ["2023-10-06", "2023-10-09", "2023-10-12"]
}

# -----------------------
# Logger info
# -----------------------
LOGGERS = [6, 9, 10, 11, 12]

LOCATION_NAMES = [
    "Lake", "Lake", "Savannah", "Wetgrassland", "Forest", "Wetgrassland",
    "Forest", "Lake", "Savannah", "Forest", "Lake", "Lagoon"
]


# -----------------------
# Output directory
# -----------------------
OUTPUT_DIR = ".../"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_sunrise_sunset(latitude, longitude, date):
    """
    Compute sunrise and sunset times for a given date and location.

    Returns
    -------
    sunrise : datetime
    sunset  : datetime
    """

    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = date

    sunrise = observer.next_rising(ephem.Sun()).datetime() + timedelta(hours=2)
    sunset = observer.next_setting(ephem.Sun()).datetime() + timedelta(hours=2)

    return sunrise, sunset


# ---------------------------------------------------------------------------

def compute_averaged_spectrum(audio_path, sr, n_fft, hop_length, n_bins_out):
    """
    Compute averaged spectrum using STFT then reduce to n_bins_out bins.

    Steps
    -----
    1. STFT (Hann window)
    2. magnitude
    3. average over time
    4. frequency grouping (dimension reduction)
    5. log transform

    Returns
    -------
    np.array shape (n_bins_out,)
    """

    if not os.path.exists(audio_path):
        return np.full(n_bins_out, np.nan)

    # Load audio
    signal, _ = librosa.load(audio_path, sr=sr)

    hop_samples = int(hop_length * sr)

    # STFT
    stft = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_samples,
        window="hann"
    )

    magnitude = np.abs(stft)

    # --------------------------------------------------
    # Temporal averaging (one spectrum per 1-min segment)
    # --------------------------------------------------
    spectrum = np.mean(magnitude, axis=1)  # (4801 bins)

    # --------------------------------------------------
    # Frequency reduction → 256 bins
    # --------------------------------------------------
    bins_per_group = len(spectrum) // n_bins_out

    spectrum = spectrum[:n_bins_out * bins_per_group]

    reduced = spectrum.reshape(n_bins_out, bins_per_group).mean(axis=1)

    # --------------------------------------------------
    # Log scaling
    # --------------------------------------------------
    log_offset = 1e-10
    reduced_db = np.log(reduced + log_offset)

    return reduced_db


# ---------------------------------------------------------------------------

def parse_filename_info(filename, month_name, latitude, longitude):
    """
    Extract metadata from filename (YYYYMMDD_HHMMSS.WAV)
    """

    date_str = (
        f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]} "
        f"{filename[9:11]}:{filename[11:13]}:{filename[13:15]}"
    )

    file_datetime = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

    file_time_int = (
        int(filename[9:11]) * 10000 +
        int(filename[11:13]) * 100 +
        int(filename[13:15])
    )

    sunrise, sunset = compute_sunrise_sunset(latitude, longitude, file_datetime)

    sunrise_int = int(sunrise.strftime("%H%M%S"))
    sunset_int = int(sunset.strftime("%H%M%S"))

    period = "day" if sunrise_int <= file_time_int <= sunset_int else "night"

    month_label = month_name.split('_')[1].upper()

    return {
        "date": file_datetime.strftime("%Y-%m-%d"),
        "time": file_datetime.strftime("%H:%M:%S"),
        "month": month_label,
        "period": period
    }


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_campaign_logger(campaign_name, base_path, logger_id, selected_dates):
    """
    Process all WAV files for one campaign + logger.
    Returns a DataFrame with metadata + spectral features.
    """

    logger_label = f"L{logger_id}"
    segment_path = os.path.join(base_path, f"{logger_id}_segments/")

    if not os.path.exists(segment_path):
        print(f"⚠️ Directory not found: {segment_path}")
        return None

    location_name = LOCATION_NAMES[logger_id - 1]

    all_files = [f for f in os.listdir(segment_path) if f.endswith(".WAV")]

    selected_files = [
        f for f in all_files
        if f"{f[:4]}-{f[4:6]}-{f[6:8]}" in selected_dates
    ]

    if not selected_files:
        return None

    selected_files.sort()

    data = []

    for filename in tqdm(selected_files, desc=f"{campaign_name} - {logger_label}"):

        file_path = os.path.join(segment_path, filename)

        metadata = parse_filename_info(
            filename, campaign_name, LATITUDE, LONGITUDE
        )

        spectrum = compute_averaged_spectrum(
            file_path, SR, N_FFT, HOP_LENGTH, N_BINS
        )

        record = {
            "file_name": filename,
            "location": location_name,
            "date": metadata["date"],
            "time": metadata["time"],
            "month": metadata["month"],
            "period": metadata["period"]
        }

        for i, value in enumerate(spectrum):
            record[f"freq_bin_{i}"] = value

        data.append(record)

    df = pd.DataFrame(data)

    encoder = LabelEncoder()
    df["date_encoded"] = encoder.fit_transform(df["date"])

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("ACOUSTIC FEATURE EXTRACTION (256 BIN VERSION)")
    print("=" * 70)

    for campaign_name, base_path in CAMPAIGN_PATHS.items():

        if campaign_name not in SELECTED_DATES:
            continue

        print(f"\nProcessing campaign: {campaign_name}")

        selected_dates = SELECTED_DATES[campaign_name]
        month_label = campaign_name.split('_')[1].upper()

        for logger_id in LOGGERS:

            df = process_campaign_logger(
                campaign_name,
                base_path,
                logger_id,
                selected_dates
            )

            if df is None or len(df) == 0:
                continue

            output_file = os.path.join(
                OUTPUT_DIR,
                f"features_{month_label}_L{logger_id}.csv"
            )

            df.to_csv(output_file, index=False)

            print(f"✅ Saved: {output_file} ({len(df)} samples)")

    print("\nProcessing complete.")


# ============================================================================
if __name__ == "__main__":
    main()
