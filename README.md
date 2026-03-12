# soundscape-mozambique-ultrasound
This repository contains all scripts and pre-processed data required to reproduce the ecoacoustic analyses and figures presented in:  "The Missing Soundscape: Ultrasound Reveals Hidden Dimensions of Ecosystem Dynamics"

# Acoustic Analysis and Classification Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

This repository contains all scripts and pre-processed data required to reproduce the ecoacoustic analyses and figures presented in:

> **"The Missing Soundscape: Ultrasound Reveals Hidden Dimensions of Ecosystem Dynamics"**
> *[Margaux Thieury, Frédéric Sèbe, Jérémy Rouch, Bamdad Sabbagh, Nicolas Grimault, Rémi Emonet, Nicolas Mathevona, Paulo Fonseca]* — submitted to *Proceedings of the National Academy of Sciences (PNAS)*

---

## Data Availability

Raw audio recordings are not distributed here due to size constraints. Selected audio samples and all intermediate data products are available on Zenodo:

**Dataset:** [https://doi.org/10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

The audio subset consists of one minute of recording every two hours for one full day per month (March, May, July, October, and December) across five habitats (wet grassland, forest, lake, pond, and savanna). These excerpts illustrate the soundscape conditions used for acoustic feature extraction and subsequent analyses.

All intermediate products (acoustic features and classification outputs) are provided on Zenodo to ensure full reproducibility of the workflow and figures without re-running the full pipeline.


---

## Repository Structure

```
soundscape-mozambique-pnas/
├── scripts/
│   ├── figure2B_radar.py          # Radar plots of dominant taxa presence
│   ├── 1-features_extraction.py   # Acoustic feature extraction
│   ├── 2-classification.py        # SVM classification
│   └── 3-Result-figure3-4-5.py    # Results and figure generation
├── data/
│   └── presence_rate_acoustic_data.xlsx
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Pipeline

The pipeline is organized into two main components: presence analysis (Figure 2B) and acoustic classification (Figures 3–5).

### A — Figure 2B: Radar plots of dominant taxa presence

- **Script:** `figure2B_radar.py`
- **Input (provided):** `data/presence_rate_acoustic_data.xlsx`
- **Description:** Generates radar (polar) plots representing the mean daily presence rate (proportion of minutes containing vocalizations) of dominant taxa across five habitats, day and night periods, and five months. Outputs one radar plot per habitat × period combination.

### B — Acoustic classification workflow

#### B.1 — Acoustic feature extraction

- **Script:** `1-features_extraction.py`
- **Output (provided on Zenodo):** `data/features_256/`
- **Description:** Extracts FFT-based spectral features from recordings, including frequency-bin energy features and band-specific representations (audible, ultrasonic, full spectrum). These features are used as input for machine learning classification.

#### B.2 — SVM classification of acoustic features

- **Script:** `2-classification.py`
- **Output (provided on Zenodo):** `results_classification/`
- **Description:** Performs supervised classification using Support Vector Machines (SVM) with training/testing splits across temporal or habitat conditions, predictions per frequency band, and probability outputs per class. Generates CSV files containing predicted labels and performance metrics.

#### B.3 — Classification results and figure generation

- **Script:** `3-Result-figure3-4-5.py`
- **Input (provided on Zenodo):** `results_classification/`
- **Description:** Computes evaluation metrics and produces visualizations including accuracy and F1-score summaries, mutual information metrics, and boxplots. Used to generate Figures 3, 4, and 5.

---

## Reproducing the Figures

All scripts can run independently. Intermediate outputs provided on Zenodo allow direct reproduction of all figures without re-running feature extraction or classification.

| Figure | Script | Zenodo data needed |
|---|---|---|
| Figure 2B | `figure2B_radar.py` | No (data included here) |
| Figures 3–5 | `3-Result-figure3-4-5.py` | Yes (`results_classification/`) |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

> [Author names] ([Year]). *The Missing Soundscape: Ultrasound Reveals Hidden Dimensions of Ecosystem Dynamics*. Proceedings of the National Academy of Sciences. https://doi.org/[ARTICLE DOI]

---

## Contact

**[Your Name]** — [your.email@institution.edu]
[ORCID: https://orcid.org/XXXX-XXXX-XXXX-XXXX]
