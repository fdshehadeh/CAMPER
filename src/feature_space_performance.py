"""
Script to compare RF and CAMPER performance across different biophysical feature spaces.
Shows where each method excels based on peptide characteristics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Import custom modules
from data_preprocessing import load_and_process_mic_data
from feature_extraction_enhanced import get_features
from BiophysicalPeptideScorer import BiophysicalPeptideScorer

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTER_TEST_SIZE = 0.2
RANDOM_STATE = 42
N_BINS = 5  # Number of bins for each feature

filepath = "../data/"
output_path = "../output/"
cache_path = "../cache/"
os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_metrics_for_subset(y_true, y_pred_rf, y_pred_camper, threshold=0.5):
    """
    Calculate metrics for a subset of predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_rf : array-like
        RF predictions
    y_pred_camper : array-like
        CAMPER predictions
    threshold : float
        Threshold for binary classification
        
    Returns:
    --------
    dict
        Dictionary with metrics for both models
    """
    rf_binary = (y_pred_rf >= threshold).astype(int)
    camper_binary = (y_pred_camper >= threshold).astype(int)
    
    # Handle cases with no predictions
    if np.sum(rf_binary) == 0:
        rf_precision = 0
    else:
        rf_precision = precision_score(y_true, rf_binary, zero_division=0)
    
    if np.sum(camper_binary) == 0:
        camper_precision = 0
    else:
        camper_precision = precision_score(y_true, camper_binary, zero_division=0)
    
    rf_recall = recall_score(y_true, rf_binary, zero_division=0)
    camper_recall = recall_score(y_true, camper_binary, zero_division=0)
    
    rf_f1 = f1_score(y_true, rf_binary, zero_division=0)
    camper_f1 = f1_score(y_true, camper_binary, zero_division=0)
    
    return {
        'rf_precision': rf_precision,
        'camper_precision': camper_precision,
        'rf_recall': rf_recall,
        'camper_recall': camper_recall,
        'rf_f1': rf_f1,
        'camper_f1': camper_f1,
        'n_samples': len(y_true),
        'n_actives': np.sum(y_true)
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 80)
    print("LOADING DATA AND FEATURES")
    print("=" * 80)
    
    # Load data
    MICfile = os.path.join(filepath, "staph_MICs_2024_10_14_updated.xlsx")
    df, labels = load_and_process_mic_data(MICfile)
    
    # Split into train/test
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=OUTER_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    df_train = df.iloc[train_indices].copy().reset_index(drop=True)
    df_test = df.iloc[test_indices].copy().reset_index(drop=True)
    y_train = labels.iloc[train_indices].copy().reset_index(drop=True)
    y_test = labels.iloc[test_indices].copy().reset_index(drop=True)
    
    print(f"Training set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")
    
    # Load or compute features
    cache_file_train = os.path.join(cache_path, "X_train_rdkit.parquet")
    if os.path.exists(cache_file_train):
        print("Loading cached training features...")
        X_train = pd.read_parquet(cache_file_train)
    else:
        print("Computing training features...")
        X_train = get_features(df_train, method='rdkit', batch_size=1000)
        X_train.to_parquet(cache_file_train)
    
    cache_file_test = os.path.join(cache_path, "X_test_rdkit.parquet")
    if os.path.exists(cache_file_test):
        print("Loading cached test features...")
        X_test = pd.read_parquet(cache_file_test)
    else:
        print("Computing test features...")
        X_test = get_features(df_test, method='rdkit', batch_size=1000)
        X_test.to_parquet(cache_file_test)
    
    print(f"Test features shape: {X_test.shape}")
    
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=1200,
        max_depth=None,
        max_features='sqrt',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("✓ Random Forest trained")
    
    # Train BiophysicalPeptideScorer (CAMPER)
    print("\nTraining BiophysicalPeptideScorer (CAMPER)...")
    scorer = BiophysicalPeptideScorer(
        gravy_column='gravy',
        helicity_column='helicity',
        charge_column='charge',
        hydrophobic_moment_column='hydrophobic_moment',
        normalize_properties=False
    )
    scorer.fit(X_train)
    print("✓ BiophysicalPeptideScorer fitted")
    
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    # RF Predictions
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # CAMPER Predictions
    _, camper_scores = scorer.predict(X_test, rf_model)
    camper_proba = camper_scores['final_score'].values
    
    # =========================================================================
    # ANALYZE PERFORMANCE ACROSS FEATURE SPACE
    # =========================================================================
    print("\nAnalyzing performance across biophysical feature space...")
    
    biophys_features = ['gravy', 'helicity', 'charge', 'hydrophobic_moment']
    feature_labels = {
        'gravy': 'GRAVY (Kyte-Doolittle scale)',
        'helicity': 'Helicity (fraction)',
        'charge': 'Charge (net units)',
        'hydrophobic_moment': 'Hydrophobic moment (4)'
    }
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    all_results = []
    
    for feature_idx, feature in enumerate(biophys_features):
        print(f"\n  Processing {feature}...")
        ax = axes[feature_idx]
        
        # Get feature values
        feature_vals = X_test[feature].values
        
        # Create equal-width bins
        min_val = np.min(feature_vals)
        max_val = np.max(feature_vals)
        bins = np.linspace(min_val, max_val, N_BINS + 1)
        bin_indices = np.digitize(feature_vals, bins) - 1
        bin_indices = np.clip(bin_indices, 0, N_BINS - 1)
        
        # Calculate metrics for each bin
        rf_precisions = []
        camper_precisions = []
        rf_recalls = []
        camper_recalls = []
        bin_ranges = []
        bin_centers = []
        rf_tp_list = []
        rf_fp_list = []
        camper_tp_list = []
        camper_fp_list = []
        
        for bin_num in range(N_BINS):
            mask = bin_indices == bin_num
            
            if np.sum(mask) == 0:
                continue
            
            y_bin = y_test[mask]
            rf_pred_bin = rf_proba[mask]
            camper_pred_bin = camper_proba[mask]
            
            metrics = get_metrics_for_subset(y_bin, rf_pred_bin, camper_pred_bin)
            
            # Calculate TP and FP at 0.5 threshold
            rf_binary = (rf_pred_bin >= 0.5).astype(int)
            camper_binary = (camper_pred_bin >= 0.5).astype(int)
            
            rf_tp = np.sum((rf_binary == 1) & (y_bin == 1))
            rf_fp = np.sum((rf_binary == 1) & (y_bin == 0))
            camper_tp = np.sum((camper_binary == 1) & (y_bin == 1))
            camper_fp = np.sum((camper_binary == 1) & (y_bin == 0))
            
            rf_tp_list.append(rf_tp)
            rf_fp_list.append(rf_fp)
            camper_tp_list.append(camper_tp)
            camper_fp_list.append(camper_fp)
            
            rf_precisions.append(metrics['rf_precision'])
            camper_precisions.append(metrics['camper_precision'])
            rf_recalls.append(metrics['rf_recall'])
            camper_recalls.append(metrics['camper_recall'])
            
            bin_range = f"{bins[bin_num]:.2f}-{bins[bin_num + 1]:.2f}"
            bin_ranges.append(bin_range)
            bin_centers.append((bins[bin_num] + bins[bin_num + 1]) / 2)
            
            all_results.append({
                'Feature': feature,
                'Bin': bin_num + 1,
                'Bin_Range': bin_range,
                'Bin_Center': bin_centers[-1],
                'N_Samples': metrics['n_samples'],
                'N_Actives': metrics['n_actives'],
                'RF_Precision': metrics['rf_precision'],
                'CAMPER_Precision': metrics['camper_precision'],
                'RF_Recall': metrics['rf_recall'],
                'CAMPER_Recall': metrics['camper_recall'],
                'RF_F1': metrics['rf_f1'],
                'CAMPER_F1': metrics['camper_f1'],
                'CAMPER_Better': metrics['camper_precision'] > metrics['rf_precision']
            })
        
        # Create comparison plot
        x_pos = np.arange(len(rf_precisions))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, rf_precisions, width, label='Random Forest', 
                       color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x_pos + width/2, camper_precisions, width, label='CAMPER', 
                       color='#55A868', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels and TP/FP counts on bars
        for i, (rf_p, camper_p) in enumerate(zip(rf_precisions, camper_precisions)):
            # RF bar - precision value
            ax.text(i - width/2, rf_p + 0.02, f'{rf_p:.2f}', 
                    ha='center', va='bottom', fontsize=9, weight='bold')
            # RF bar - TP/FP counts
            ax.text(i - width/2, rf_p + 0.08, f'TP:{int(rf_tp_list[i])}\nFP:{int(rf_fp_list[i])}', 
                    ha='center', va='bottom', fontsize=7)
            
            # CAMPER bar - precision value
            ax.text(i + width/2, camper_p + 0.02, f'{camper_p:.2f}', 
                    ha='center', va='bottom', fontsize=9, weight='bold')
            # CAMPER bar - TP/FP counts
            ax.text(i + width/2, camper_p + 0.08, f'TP:{int(camper_tp_list[i])}\nFP:{int(camper_fp_list[i])}', 
                    ha='center', va='bottom', fontsize=7)
        
        # Customize subplot
        ax.set_xlabel(feature_labels[feature], fontsize=11, weight='bold')
        ax.set_ylabel('Precision', fontsize=11, weight='bold')
        ax.set_title(feature_labels[feature], fontsize=12, weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_ranges, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Method Performance Across Biophysical Feature Space\n(TP = True Positives, FP = False Positives)', 
                 fontsize=15, weight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, 'feature_space_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'feature_space_performance.pdf'), 
                bbox_inches='tight')
    print("\n✓ Saved: feature_space_performance.png / pdf")
    plt.close()
    
    # =========================================================================
    # SUMMARY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("FEATURE SPACE PERFORMANCE SUMMARY")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    for feature in biophys_features:
        feature_data = results_df[results_df['Feature'] == feature]
        
        print(f"\n{feature_labels[feature]}:")
        print(f"{'Bin':<8} {'Range':<20} {'Samples':<10} {'Actives':<8} {'RF Prec':<10} {'CAMPER Prec':<12} {'Winner':<12}")
        print("-" * 80)
        
        camper_better_count = 0
        for _, row in feature_data.iterrows():
            winner = 'CAMPER ✓' if row['CAMPER_Better'] else 'RF ✓'
            if row['CAMPER_Better']:
                camper_better_count += 1
            
            print(f"{int(row['Bin']):<8} {row['Bin_Range']:<20} {int(row['N_Samples']):<10} "
                  f"{int(row['N_Actives']):<8} {row['RF_Precision']:.3f}      "
                  f"{row['CAMPER_Precision']:.3f}        {winner:<12}")
        
        print(f"\n  → CAMPER better in {camper_better_count}/{len(feature_data)} bins "
              f"({camper_better_count/len(feature_data)*100:.0f}%)")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS")
    print("=" * 80)
    
    camper_wins = results_df['CAMPER_Better'].sum()
    total_bins = len(results_df)
    
    print(f"\nCAMPER outperforms RF in {camper_wins}/{total_bins} feature-space bins ({camper_wins/total_bins*100:.1f}%)")
    
    # Find where CAMPER is strongest
    camper_better_data = results_df[results_df['CAMPER_Better']]
    if len(camper_better_data) > 0:
        print("\nRegions where CAMPER excels:")
        top_camper = camper_better_data.nlargest(5, 'CAMPER_Precision')[
            ['Feature', 'Bin_Range', 'CAMPER_Precision', 'RF_Precision', 'N_Actives']
        ]
        for _, row in top_camper.iterrows():
            improvement = row['CAMPER_Precision'] - row['RF_Precision']
            print(f"  • {row['Feature'].upper()}: {row['Bin_Range']:<15} "
                  f"CAMPER {row['CAMPER_Precision']:.3f} vs RF {row['RF_Precision']:.3f} "
                  f"({improvement:+.3f}) - {int(row['N_Actives'])} actives")
    
    # Find where RF is strongest
    rf_better_data = results_df[~results_df['CAMPER_Better']]
    if len(rf_better_data) > 0:
        print("\nRegions where RF excels:")
        top_rf = rf_better_data.nlargest(5, 'RF_Precision')[
            ['Feature', 'Bin_Range', 'RF_Precision', 'CAMPER_Precision', 'N_Actives']
        ]
        for _, row in top_rf.iterrows():
            improvement = row['RF_Precision'] - row['CAMPER_Precision']
            print(f"  • {row['Feature'].upper()}: {row['Bin_Range']:<15} "
                  f"RF {row['RF_Precision']:.3f} vs CAMPER {row['CAMPER_Precision']:.3f} "
                  f"({improvement:+.3f}) - {int(row['N_Actives'])} actives")
    
    print("\n" + "=" * 80)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_path, 'feature_space_performance.csv'), index=False)
    print("✓ Saved: feature_space_performance.csv")


if __name__ == '__main__':
    main()
