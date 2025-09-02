# Hybrid Molecular Representations

This repository contains the implementation for our ICDM REU Symposium 2025 paper:

**"Hybrid Molecular Representations: Combining Descriptors and Fingerprints for Robust Property Prediction in Low-Data Regimes"**

We present the first systematic study of hybrid molecular features that fuse interpretable descriptors with expressive ECFP fingerprints. Through experiments on classification (BBBP) and regression (ESOL, FreeSolv) tasks, we show that hybrid representations consistently outperform single-source baselines in both low-data and full-data settings.

## Overview

- Combines RDKit-based molecular descriptors with ECFP fingerprints
- Evaluates performance across:
  - **BBBP** (Blood-Brain Barrier Penetration) — classification
  - **ESOL** (Aqueous Solubility) — regression
  - **FreeSolv** (Hydration Free Energy) — regression
- Models used: Random Forest (classification and regression variants)
- Evaluation metrics:
  - ROC-AUC (for classification)
  - RMSE (for regression)

## Repository Structure
```bash
hybrid-molecular-representations/
├── hybrid_model.py         # Core script for training and evaluation
├── requirements.txt        # Required Python packages
└── README.md               # Project description and instructions
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Hybrid-Molecular-Representations.git
cd Hybrid-Molecular-Representations
```

### 2. Install Dependencies

We recommend using a virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Script

Run the main script to train and evaluate hybrid models:

```bash
python hybrid_model.py
```

You can adjust the dataset loading code inside hybrid_model.py to test individual datasets like BBBP, ESOL, or FreeSolv.

## Reproducibility

All datasets are loaded via DeepChem, and hybrid features are generated using RDKit descriptors + ECFP fingerprints. The full preprocessing pipeline and evaluation procedure are included in the script.

If you wish to replicate results in a notebook environment, the core logic is modular and can be easily adapted.

## Citation

If you find this work useful, please cite:

Alex Liu, "Hybrid Molecular Representations: Combining Descriptors and Fingerprints for Robust Property Prediction in Low-Data Regimes", REU Symposium at IEEE ICDM 2025.

Contact

For questions or collaboration inquiries, please contact:

Alex Liu / ALiu7@nd.edu
