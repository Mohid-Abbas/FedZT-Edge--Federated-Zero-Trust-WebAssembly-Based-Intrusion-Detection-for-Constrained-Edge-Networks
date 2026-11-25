"""
Preprocessing and Dirichlet Non-IID Partitioning (Steps 3 & 4)
Implements cleaning, identifier removal, imputation, Chi2 top-k feature selection,
StandardScaler normalization, majority-to-minority balancing (cap 2:1),
and parameterized Dirichlet Non-IID shard generator for FedZT-Edge.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
import joblib

try:
    from .config import PROCESSED_DATA_DIR, RAW_DATA_DIR
except (ImportError, ValueError):
    from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

IDENTIFIER_COLS = [
    'id', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip',
    'sport', 'dport', 'src_port', 'dst_port', 'Destination Port',
    'timestamp', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp'
]

def clean_and_normalize_dataset(dataset_name="UNSW-NB15", k_features=20, random_state=42):
    """
    Cleans, imputes, selects top-k features via Chi-square, balances classes at 2:1 max ratio,
    and produces 80/20 train/test splits.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    if dataset_name == "UNSW-NB15":
        raw_path = os.path.join(RAW_DATA_DIR, "unsw_nb15_raw.parquet")
        df = pd.read_parquet(raw_path)
        label_col = 'label'
        # Drop categorical text attack category if present for numerical pipeline
        if 'attack_cat' in df.columns:
            attack_cats = df['attack_cat']
            df = df.drop(columns=['attack_cat'])
        else:
            attack_cats = df[label_col].astype(str)
    elif dataset_name == "CICIDS2017":
        raw_path = os.path.join(RAW_DATA_DIR, "cicids2017_raw.csv")
        df = pd.read_csv(raw_path)
        # Determine label column
        if 'binary_label' in df.columns:
            label_col = 'binary_label'
        else:
            label_candidates = [c for c in df.columns if 'label' in c.lower()]
            orig_label_col = label_candidates[0]
            df['binary_label'] = (df[orig_label_col] != 'BENIGN').astype(int)
            label_col = 'binary_label'
            df = df.drop(columns=[orig_label_col])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    print(f"[{dataset_name}] Raw shape: {df.shape}")
    
    # 1. Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Drop identifier columns
    cols_to_drop = [c for c in df.columns if c in IDENTIFIER_COLS or any(id_col.lower() in c.lower() for id_col in ['ip', 'port', 'timestamp', 'flow id'])]
    # Ensure label column is preserved
    cols_to_drop = [c for c in cols_to_drop if c != label_col]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Separate features and target
    y = df[label_col].values.astype(int)
    X_df = df.drop(columns=[label_col])
    
    # Handle infinite or NaN values in numerical columns
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    
    # Handle categorical columns with One-Hot Encoding
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)
        
    # 3. Imputation (median)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_df)
    feature_names = np.array(X_df.columns)
    
    # 4. Remove zero-variance columns
    variances = np.var(X_imputed, axis=0)
    non_zero_idx = np.where(variances > 1e-6)[0]
    X_filtered = X_imputed[:, non_zero_idx]
    feature_names = feature_names[non_zero_idx]
    
    # 5. Cap majority-to-minority balancing at 2:1
    benign_indices = np.where(y == 0)[0]
    attack_indices = np.where(y == 1)[0]
    n_min = min(len(benign_indices), len(attack_indices))
    n_max_allowed = int(2.0 * n_min)
    
    np.random.seed(random_state)
    if len(benign_indices) > n_max_allowed:
        benign_indices = np.random.choice(benign_indices, size=n_max_allowed, replace=False)
    if len(attack_indices) > n_max_allowed:
        attack_indices = np.random.choice(attack_indices, size=n_max_allowed, replace=False)
        
    keep_indices = np.sort(np.concatenate([benign_indices, attack_indices]))
    X_balanced = X_filtered[keep_indices]
    y_balanced = y[keep_indices]
    
    # 6. Feature Selection via Chi-Square (Step 4)
    # Ensure non-negative for chi2
    X_non_neg = X_balanced - np.min(X_balanced, axis=0)
    k_actual = min(k_features, X_balanced.shape[1])
    selector = SelectKBest(score_func=chi2, k=k_actual)
    selector.fit(X_non_neg, y_balanced)
    
    selected_indices = np.argsort(selector.scores_)[::-1][:k_actual]
    selected_feature_names = feature_names[selected_indices].tolist()
    chi2_scores = selector.scores_[selected_indices].tolist()
    
    X_selected = X_balanced[:, selected_indices]
    
    # 7. Train/Test Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_balanced, test_size=0.20, random_state=random_state, stratify=y_balanced
    )
    
    # 8. StandardScaler normalization (fit on train, transform test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    processed_pkg = {
        "dataset_name": dataset_name,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": selected_feature_names,
        "chi2_scores": chi2_scores,
        "k_features": k_actual,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_class_balance": {
            "benign": int((y_train == 0).sum()),
            "attack": int((y_train == 1).sum())
        }
    }
    
    save_path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name.lower().replace('-', '_')}_k{k_actual}.joblib")
    joblib.dump(processed_pkg, save_path)
    print(f"[{dataset_name}] Processed: Train={X_train.shape}, Test={X_test.shape}, Saved to {save_path}")
    
    return processed_pkg

def partition_dirichlet_non_iid(X, y, num_clients=10, alpha=0.5, seed=42):
    """
    Partitions data across num_clients using a Dirichlet distribution over classes.
    alpha -> infinity produces IID partitioning.
    alpha -> 0.1 produces severe non-IID skew.
    Returns: dict {client_id: (X_client, y_client)}
    """
    np.random.seed(seed)
    num_classes = len(np.unique(y))
    client_indices = [[] for _ in range(num_clients)]
    
    if alpha is None or alpha == float('inf'):
        # Uniform IID Partition
        indices = np.random.permutation(len(y))
        splits = np.array_split(indices, num_clients)
        client_shards = {}
        for c in range(num_clients):
            c_idx = splits[c]
            client_shards[c] = (X[c_idx], y[c_idx])
        return client_shards
        
    for k in range(num_classes):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)
        
        # Draw proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Normalize to prevent rounding issues
        proportions = proportions / proportions.sum()
        
        # Partition indices according to proportions
        counts = (proportions * len(idx_k)).astype(int)
        # Allocate any remainder
        remainder = len(idx_k) - counts.sum()
        for i in range(remainder):
            counts[i % num_clients] += 1
            
        splits = np.split(idx_k, np.cumsum(counts)[:-1])
        for c in range(num_clients):
            client_indices[c].extend(splits[c])
            
    client_shards = {}
    for c in range(num_clients):
        c_idx = np.array(client_indices[c])
        np.random.shuffle(c_idx)
        # Ensure at least minimal samples per client
        if len(c_idx) == 0:
            c_idx = np.random.choice(len(y), size=10, replace=False)
        client_shards[c] = (X[c_idx], y[c_idx])
        
    return client_shards

def analyze_feature_intersection(unsw_pkg, cic_pkg):
    """
    Step 4: Union/intersection analysis of selected features across datasets.
    """
    unsw_feats = set(unsw_pkg['feature_names'])
    cic_feats = set(cic_pkg['feature_names'])
    intersection = unsw_feats.intersection(cic_feats)
    union = unsw_feats.union(cic_feats)
    
    report = {
        "unsw_count": len(unsw_feats),
        "cic_count": len(cic_feats),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard_similarity": len(intersection) / len(union) if union else 0.0,
        "shared_features": list(intersection)
    }
    return report

if __name__ == "__main__":
    print("Testing Preprocessing Pipeline...")
    unsw = clean_and_normalize_dataset("UNSW-NB15", k_features=20)
    cic = clean_and_normalize_dataset("CICIDS2017", k_features=20)
    inter = analyze_feature_intersection(unsw, cic)
    print("Feature analysis:", json.dumps(inter, indent=2))
