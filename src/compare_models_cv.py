"""
Script to compare ROC and PR curves for:
1. Homology-aware CV + Random Forest
2. Random split CV + Random Forest
3. Homology-aware CV + BiophysicalPeptideScorer (CAMPER)
4. Random split CV + BiophysicalPeptideScorer (CAMPER)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Import custom modules
from data_preprocessing import load_and_process_mic_data
from feature_extraction_enhanced import get_features
from matrix_based_split import SimilarityMatrix
from cv_utilities import HomologyAwareCV, RandomSplitCV, run_cv_experiment

# =============================================================================
# CONFIGURATION
# =============================================================================
SIMILARITY_THRESHOLD = 0.80
OUTER_TEST_SIZE = 0.2
RANDOM_STATE = 42
K_FOLDS = 5

filepath = "../data/"
output_path = "../output/"
cache_path = "../cache/"
os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)


# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 80)
    print("LOADING DATA AND FEATURES")
    print("=" * 80)
    
    # Load data
    MICfile = os.path.join(filepath, "staph_MICs_2024_10_14_updated.xlsx")
    df, labels = load_and_process_mic_data(MICfile)
    
    # Load or create similarity matrix
    matrix_cache_file = os.path.join(cache_path, "similarity_matrix_staph.npz")
    if os.path.exists(matrix_cache_file):
        print("Loading cached similarity matrix...")
        similarity_matrix = SimilarityMatrix.load(matrix_cache_file)
    else:
        print("Computing similarity matrix...")
        similarity_matrix = SimilarityMatrix(df['sequence'].values)
        similarity_matrix.save(matrix_cache_file)
    
    # Split into train/test
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=OUTER_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    df_train = df.iloc[train_indices].copy().reset_index(drop=True)
    y_train = labels.iloc[train_indices].copy().reset_index(drop=True)
    
    print(f"Training set size: {len(df_train)}")
    
    # Load or compute features
    cache_file = os.path.join(cache_path, "X_train_rdkit.parquet")
    if os.path.exists(cache_file):
        print("Loading cached features...")
        features = pd.read_parquet(cache_file)
    else:
        print("Computing features...")
        features = get_features(df_train, method='rdkit', batch_size=1000)
        features.to_parquet(cache_file)
    
    print(f"Feature shape: {features.shape}")
    
    print("\n" + "=" * 80)
    print("RUNNING CROSS-VALIDATION EXPERIMENTS")
    print("=" * 80)
    
    results = {}
    
    # =========================================================================
    # Experiment 1: Homology-aware CV + Random Forest
    # =========================================================================
    print("\n[1/4] Homology-aware CV + Random Forest")
    print("-" * 80)
    homology_cv = HomologyAwareCV(
        similarity_matrix=similarity_matrix,
        original_indices=train_indices,
        threshold=SIMILARITY_THRESHOLD,
        n_splits=K_FOLDS,
        random_state=RANDOM_STATE
    )
    results['Homology split RF'] = run_cv_experiment(features, y_train, homology_cv, 'rf')
    
    # =========================================================================
    # Experiment 2: Random split CV + Random Forest
    # =========================================================================
    print("\n[2/4] Random split CV + Random Forest")
    print("-" * 80)
    random_cv = RandomSplitCV(n_splits=K_FOLDS, random_state=RANDOM_STATE)
    results['Random split RF'] = run_cv_experiment(features, y_train, random_cv, 'rf')
    
    # =========================================================================
    # Experiment 3: Homology-aware CV + CAMPER
    # =========================================================================
    print("\n[3/4] Homology-aware CV + BiophysicalPeptideScorer (CAMPER)")
    print("-" * 80)
    results['Homology split CAMPER'] = run_cv_experiment(
        features, y_train, homology_cv, 'camper'
    )
    
    # =========================================================================
    # Experiment 4: Random split CV + CAMPER
    # =========================================================================
    print("\n[4/4] Random split CV + BiophysicalPeptideScorer (CAMPER)")
    print("-" * 80)
    results['Random split CAMPER'] = run_cv_experiment(features, y_train, random_cv, 'camper')
    
    # =========================================================================
    # PLOT RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    # Define colors and line styles for each method
    colors = {
        'Homology split RF': '#1f77b4',
        'Random split RF': '#ff7f0e',
        'Homology split CAMPER': '#2ca02c',
        'Random split CAMPER': '#d62728'
    }
    
    linestyles = {
        'Homology split RF': '-',
        'Random split RF': '--',
        'Homology split CAMPER': '-',
        'Random split CAMPER': '--'
    }
    
    # =========================================================================
    # ROC Curve Comparison
    # =========================================================================
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    
    for method_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred'])
        auc = roc_auc_score(result['y_true'], result['y_pred'])
        
        ax_roc.plot(
            fpr, tpr,
            label=f"{method_name} (AUC = {auc:.3f})",
            color=colors[method_name],
            linestyle=linestyles[method_name],
            linewidth=2.5,
            alpha=0.8
        )
    
    # Add diagonal (random chance)
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5, alpha=0.5)
    
    ax_roc.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=12, weight='bold')
    ax_roc.set_title('ROC Curve Comparison: Homology-aware vs Random Split CV\nwith RF vs CAMPER', 
                     fontsize=14, weight='bold')
    ax_roc.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'roc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'roc_comparison.pdf'), bbox_inches='tight')
    print("Saved: roc_comparison.png / pdf")
    plt.close()
    
    # =========================================================================
    # Precision-Recall Curve Comparison
    # =========================================================================
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    
    for method_name, result in results.items():
        precision, recall, _ = precision_recall_curve(result['y_true'], result['y_pred'])
        ap = average_precision_score(result['y_true'], result['y_pred'])
        
        ax_pr.plot(
            recall, precision,
            label=f"{method_name} (AP = {ap:.3f})",
            color=colors[method_name],
            linestyle=linestyles[method_name],
            linewidth=2.5,
            alpha=0.8
        )
    
    # Add baseline (no skill)
    baseline = np.mean(results['Homology split RF']['y_true'])
    ax_pr.axhline(baseline, color='k', linestyle='--', linewidth=1.5, alpha=0.5, 
                  label=f'No Skill (Baseline = {baseline:.3f})')
    
    ax_pr.set_xlabel('Recall', fontsize=12, weight='bold')
    ax_pr.set_ylabel('Precision', fontsize=12, weight='bold')
    ax_pr.set_title('Precision-Recall Curve Comparison: Homology-aware vs Random Split CV\nwith RF vs CAMPER', 
                    fontsize=14, weight='bold')
    ax_pr.legend(loc='best', fontsize=11, framealpha=0.95)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'pr_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'pr_comparison.pdf'), bbox_inches='tight')
    print("Saved: pr_comparison.png / pdf")
    plt.close()
    
    # =========================================================================
    # PRINT SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    summary_data = []
    for method_name, result in results.items():
        auc = roc_auc_score(result['y_true'], result['y_pred'])
        ap = average_precision_score(result['y_true'], result['y_pred'])
        
        summary_data.append({
            'Method': method_name,
            'Mean CV AUC': result['mean_auc'],
            'Std CV AUC': result['std_auc'],
            'Test AUC': auc,
            'Test AP': ap
        })
        
        print(f"\n{method_name}")
        print(f"  CV AUC: {result['mean_auc']:.4f} (±{result['std_auc']:.4f})")
        print(f"  Test AUC: {auc:.4f}")
        print(f"  Test AP: {ap:.4f}")
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_path, 'summary_statistics.csv'), index=False)
    print("\n" + "=" * 80)
    print(summary_df.to_string(index=False))
    print("\nSaved: summary_statistics.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()
