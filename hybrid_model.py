"""
hybrid_model.py
Author: Alex Liu
Description: Hybrid molecular property prediction using concatenated
RDKit descriptors and ECFP fingerprints. Supports classification (BBBP)
and regression (ESOL, FreeSolv) tasks using Random Forest models.

Dependencies:
- deepchem
- rdkit
- scikit-learn
- pandas
- numpy
"""

import deepchem as dc
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm

# ---------------------------------------------
# Configuration
# ---------------------------------------------
DATASET = 'BBBP'  # Options: 'BBBP', 'ESOL', 'FreeSolv'
ECFP_RADIUS = 2
ECFP_NBITS = 2048
TEST_SIZE = 0.2
RANDOM_SEED = 42
N_ESTIMATORS = 500

# ---------------------------------------------
# Feature Generation
# ---------------------------------------------

def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def generate_ecfp(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(ECFP_NBITS)
    fp = GetMorganFingerprintAsBitVect(mol, radius=ECFP_RADIUS, nBits=ECFP_NBITS)
    return np.array(fp)

def generate_descriptors(smiles):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return [0] * len(Descriptors.descList)
    return [desc_func(mol) for _, desc_func in Descriptors.descList]

def generate_hybrid_features(smiles_list):
    ecfps = []
    descs = []
    for smi in tqdm(smiles_list, desc="Generating Features"):
        ecfps.append(generate_ecfp(smi))
        descs.append(generate_descriptors(smi))
    ecfps = np.array(ecfps)
    descs = np.array(descs)
    return np.concatenate([ecfps, descs], axis=1)

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------

def load_dataset(name):
    if name == 'BBBP':
        task, (train, _), _ = dc.molnet.load_bbbp(featurizer='Raw', splitter='random')
    elif name == 'ESOL':
        task, (train, _), _ = dc.molnet.load_esol(featurizer='Raw', splitter='random')
    elif name == 'FreeSolv':
        task, (train, _), _ = dc.molnet.load_freesolv(featurizer='Raw', splitter='random')
    else:
        raise ValueError(f"Unknown dataset: {name}")
    df = pd.DataFrame({
        'smiles': train.ids,
        'label': train.y.flatten()
    })
    return df

# ---------------------------------------------
# Train + Evaluate
# ---------------------------------------------

def train_and_evaluate(X, y, task_type):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {auc:.4f}")
    else:
        model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"RMSE: {rmse:.4f}")

# ---------------------------------------------
# Main
# ---------------------------------------------

def main():
    print(f"\nLoading dataset: {DATASET}")
    df = load_dataset(DATASET)

    print(f"\nGenerating hybrid features (ECFP + Descriptors)...")
    X = generate_hybrid_features(df['smiles'].tolist())
    y = df['label'].values

    task_type = 'classification' if DATASET == 'BBBP' else 'regression'
    print(f"\nRunning {task_type} model with Random Forest...")
    train_and_evaluate(X, y, task_type)

if __name__ == "__main__":
    main()
