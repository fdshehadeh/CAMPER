# matrix_sensitivity_analysis.py

import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Import the final, robust splitting code
from matrix_based_split import SimilarityMatrix, homology_aware_split
# (Assuming your other custom modules are in the path)
from feature_extraction_enhanced import get_features
from data_preprocessing import load_and_process_mic_data

# Ignore common warnings and set plotting style
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================
filepath = "../data/"
output_path = "../output/"
cache_path = "../cache/"
os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

# =============================================================================
# DATA AND FEATURE LOADING (WITH CACHING)
# =============================================================================
print("Step 1: Loading and Preprocessing Data")
MICfile = os.path.join(filepath, "staph_MICs_2024_10_14_updated.xlsx")
df, labels = load_and_process_mic_data(MICfile)
print(f"Loaded {len(df)} peptides.")

print("\nStep 2: Managing Features")
features_cache_file = os.path.join(cache_path, "features_rdkit_biophysical.parquet")
if os.path.exists(features_cache_file):
    print(f"Loading cached features from {features_cache_file}...")
    X = pd.read_parquet(features_cache_file)
else:
    print("Extracting features (RDKit + biophysical)...")
    X = get_features(df, method='rdkit', batch_size=1000)
    X.to_parquet(features_cache_file)
print(f"Feature matrix shape: {X.shape}")

# =============================================================================
# SIMILARITY MATRIX MANAGEMENT
# =============================================================================
print("\nStep 3: Managing Similarity Matrix")
sequences = df['sequence'].values
matrix_cache_file = os.path.join(cache_path, "similarity_matrix_staph.npz")
if os.path.exists(matrix_cache_file):
    similarity_matrix = SimilarityMatrix.load(matrix_cache_file)
else:
    similarity_matrix = SimilarityMatrix(sequences)
    similarity_matrix.save(matrix_cache_file)


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================
def run_homology_sensitivity_analysis(similarity_matrix, labels, X,
                                      similarity_thresholds, random_seeds, test_size=0.2):
    """
    Runs a full sensitivity analysis using the homology-aware splitting function,
    capturing performance and structural metrics.
    """
    results = []
    print("\nStep 4: Running Homology Sensitivity Analysis")
    print(f"  Thresholds: {similarity_thresholds}")
    print(f"  Random seeds: {random_seeds}")

    for threshold in similarity_thresholds:
        for seed in random_seeds:
            print(f"\n--- Testing Threshold: {threshold:.2f}, Seed: {seed} ---")
            try:
                # Get split indices and cluster metrics from the splitting function
                train_indices, test_indices, n_clusters, largest_cluster = homology_aware_split(
                    similarity_matrix, labels,
                    test_size=test_size,
                    similarity_threshold=threshold,
                    random_state=seed,
                    verbose=False
                )

                if len(test_indices) < 10:
                    print("  Warning: Test set is too small for reliable metrics. Skipping.")
                    continue

                # Prepare data for this specific split
                X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
                y_train, y_test = labels.iloc[train_indices], labels.iloc[test_indices]

                # Train a model for this split
                model = RandomForestClassifier(
                    n_estimators=1200,
                    max_depth=None,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)

                # Evaluate the model
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                roc_auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)

                # Top 10% precision
                n_top_10 = max(1, int(len(y_test) * 0.1))
                top_10_indices = np.argsort(-y_pred_proba)[:n_top_10]
                top_10_precision = np.mean(y_test.iloc[top_10_indices])

                results.append({
                    'threshold': threshold,
                    'seed': seed,
                    'roc_auc': roc_auc,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'top_10_precision': top_10_precision,
                    'test_pct': len(test_indices) / len(labels) * 100,
                    'n_clusters': n_clusters,
                    'largest_cluster_size': largest_cluster
                })
                print(
                    f"  Split: {len(train_indices)}/{len(test_indices)} | Largest Cluster: {largest_cluster} | AUC: {roc_auc:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")

    return pd.DataFrame(results)


# --- Define parameters and run the analysis ---
similarity_thresholds = [0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]
random_seeds = range(100)
results_df = run_homology_sensitivity_analysis(similarity_matrix, labels, X, similarity_thresholds, random_seeds)

# =============================================================================
# VISUALIZATION AND RESULTS ("ELBOW METHOD")
# =============================================================================
print("\nStep 5: Aggregating and Visualizing Results")
if not results_df.empty:
    # Save detailed results to a CSV file
    results_path = os.path.join(output_path, "homology_elbow_analysis.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Full results saved to {results_path}")

    # --- UPDATED 2x2 PLOT ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    fig.suptitle('Elbow Method for Homology Threshold Selection', fontsize=16, weight='bold')

    # Plot 1: The "Elbow" - Size of Largest Cluster
    sns.lineplot(data=results_df, x='threshold', y='largest_cluster_size', marker='o', ax=axes[0, 0], errorbar='sd')
    axes[0, 0].set_title('Data Structure vs. Threshold', weight='bold')
    axes[0, 0].set_ylabel('Size of Largest Cluster')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(which='both', linestyle='--')

    # Plot 2: Model Performance (AUC)
    sns.lineplot(data=results_df, x='threshold', y='roc_auc', marker='o', ax=axes[0, 1], errorbar='sd')
    axes[0, 1].set_title('Model Performance vs. Threshold', weight='bold')
    axes[0, 1].set_ylabel('ROC AUC (Mean & Std Dev)')

    # Plot 3: Number of Clusters
    sns.lineplot(data=results_df, x='threshold', y='n_clusters', marker='o', ax=axes[1, 0], errorbar='sd')
    axes[1, 0].set_title('Number of Clusters vs. Threshold', weight='bold')
    axes[1, 0].set_ylabel('Total Clusters Found')
    axes[1, 0].set_xlabel('Similarity Threshold')  # Text removed

    # Plot 4: Top 10% Precision (Replaced Test Set Size)
    sns.lineplot(data=results_df, x='threshold', y='top_10_precision', marker='o', ax=axes[1, 1], errorbar='sd')
    axes[1, 1].set_title('Top 10% Precision vs. Threshold', weight='bold')
    axes[1, 1].set_ylabel('Precision in Top 10%')
    axes[1, 1].set_xlabel('Similarity Threshold')  # Text removed

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_path, "homology_elbow_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Analysis plot saved to {plot_path}")
    plt.show()

print("\nAnalysis complete.")
