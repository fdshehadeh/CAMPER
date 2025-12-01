"""
Script to generate precision improvement plots comparing RF and CAMPER
across bins of biophysical features (gravy, helicity, charge, hydrophobic_moment).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

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
    
    # Load or compute features for training set
    cache_file_train = os.path.join(cache_path, "X_train_rdkit.parquet")
    if os.path.exists(cache_file_train):
        print("Loading cached training features...")
        X_train = pd.read_parquet(cache_file_train)
    else:
        print("Computing training features...")
        X_train = get_features(df_train, method='rdkit', batch_size=1000)
        X_train.to_parquet(cache_file_train)
    
    print(f"Training features shape: {X_train.shape}")
    
    # Load or compute features for test set
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
    print("GENERATING PREDICTIONS ON TEST SET")
    print("=" * 80)
    
    # RF Predictions
    print("\nGenerating RF predictions...")
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    
    # CAMPER Predictions
    print("Generating CAMPER predictions...")
    _, camper_scores = scorer.predict(X_test, rf_model)
    camper_proba = camper_scores['final_score'].values
    camper_pred = (camper_proba >= 0.5).astype(int)
    
    print("\n" + "=" * 80)
    print("CALCULATING PRECISION ACROSS FEATURE BINS")
    print("=" * 80)
    
    # Biophysical features to analyze
    biophys_features = ['gravy', 'helicity', 'charge', 'hydrophobic_moment']
    
    # Custom axis labels for each feature
    feature_labels = {
        'gravy': 'GRAVY (Kyte-Doolittle scale)',
        'helicity': 'Helicity (fraction)',
        'charge': 'Charge (net units)',
        'hydrophobic_moment': 'Hydrophobic moment (μ)'
    }
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    all_results = []
    
    for feature_idx, feature in enumerate(biophys_features):
        print(f"\nProcessing {feature}...")
        ax = axes[feature_idx]
        
        # Get feature values
        feature_vals = X_test[feature].values
        
        # Create bins with equal range (not equal count)
        min_val = np.min(feature_vals)
        max_val = np.max(feature_vals)
        bins = np.linspace(min_val, max_val, N_BINS + 1)
        
        # Assign samples to bins
        bin_indices = np.digitize(feature_vals, bins) - 1
        bin_indices = np.clip(bin_indices, 0, N_BINS - 1)
        
        # Calculate precision for each bin
        rf_precisions = []
        camper_precisions = []
        improvements = []
        bin_sizes = []
        bin_ranges = []
        
        for bin_num in range(N_BINS):
            mask = bin_indices == bin_num
            
            if np.sum(mask) == 0:
                continue
            
            # Calculate precision
            y_bin = y_test[mask]
            rf_pred_bin = rf_pred[mask]
            camper_pred_bin = camper_pred[mask]
            
            # Precision (avoid division by zero)
            if np.sum(rf_pred_bin) > 0:
                rf_prec = precision_score(y_bin, rf_pred_bin, zero_division=0)
            else:
                rf_prec = 0
            
            if np.sum(camper_pred_bin) > 0:
                camper_prec = precision_score(y_bin, camper_pred_bin, zero_division=0)
            else:
                camper_prec = 0
            
            rf_precisions.append(rf_prec)
            camper_precisions.append(camper_prec)
            improvements.append(camper_prec - rf_prec)
            bin_sizes.append(np.sum(mask))
            bin_ranges.append(f"{bins[bin_num]:.2f}-{bins[bin_num + 1]:.2f}")
            
            all_results.append({
                'Feature': feature,
                'Bin': bin_num + 1,
                'Bin_Range': bin_ranges[-1],
                'Bin_Size': bin_sizes[-1],
                'RF_Precision': rf_prec,
                'CAMPER_Precision': camper_prec,
                'Improvement': camper_prec - rf_prec
            })
        
        # Create combined plot with dual y-axes
        x_pos = np.arange(len(rf_precisions))
        
        # Bar chart for improvements on primary axis
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in improvements]
        bars = ax.bar(x_pos, improvements, color=colors, alpha=0.3, edgecolor='black', linewidth=1, label='Improvement (CAMPER - RF)')
        
        # Line plots on secondary axis
        ax2 = ax.twinx()
        ax2.plot(x_pos, rf_precisions, marker='o', linewidth=2.5, markersize=8, 
                label='RF Precision', color='#4C72B0', alpha=0.9, zorder=5)
        ax2.plot(x_pos, camper_precisions, marker='s', linewidth=2.5, markersize=8, 
                label='CAMPER Precision', color='#55A868', alpha=0.9, zorder=5)
        
        # Add value labels on line points
        for i, (rf_prec, camper_prec) in enumerate(zip(rf_precisions, camper_precisions)):
            ax2.text(i, rf_prec + 0.03, f'{rf_prec:.2f}', ha='center', va='bottom', 
                    fontsize=7, weight='bold', color='#4C72B0')
            ax2.text(i, camper_prec - 0.05, f'{camper_prec:.2f}', ha='center', va='top', 
                    fontsize=7, weight='bold', color='#55A868')
        
        # Customize subplots
        ax.set_xlabel(feature_labels[feature], fontsize=11, weight='bold')
        ax.set_ylabel('Improvement (CAMPER - RF)', fontsize=10, weight='bold', color='gray')
        ax2.set_ylabel('Precision', fontsize=10, weight='bold')
        ax.set_title(feature_labels[feature], fontsize=12, weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_ranges, rotation=45, ha='right', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_ylim([0, 1.15])
        ax.grid(axis='y', alpha=0.2)
        
        # Color y-axes labels to match
        ax.tick_params(axis='y', labelcolor='gray')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
        
        # Add bin size information above x-axis
        y_max = 1.12
        for i, size in enumerate(bin_sizes):
            ax2.text(i, y_max, f'n={size}', ha='center', va='bottom', fontsize=8, 
                    color='gray', weight='bold')
    
    plt.suptitle('Precision Comparison and Improvement Across Biophysical Feature Bins\n2 Lines (Precision) + Bar Chart (Improvement)', 
                 fontsize=15, weight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, 'precision_by_feature_bins.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'precision_by_feature_bins.pdf'), 
                bbox_inches='tight')
    print("\n✓ Saved: precision_by_feature_bins.png / pdf")
    plt.close()
    
    # =========================================================================
    # SAVE DETAILED RESULTS
    # =========================================================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_path, 'precision_by_bins_detailed.csv'), index=False)
    print("✓ Saved: precision_by_bins_detailed.csv")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("PRECISION IMPROVEMENT SUMMARY")
    print("=" * 80)
    
    for feature in biophys_features:
        feature_results = results_df[results_df['Feature'] == feature]
        avg_improvement = feature_results['Improvement'].mean()
        max_improvement = feature_results['Improvement'].max()
        min_improvement = feature_results['Improvement'].min()
        
        print(f"\n{feature_labels[feature]}:")
        print(f"  Average Improvement: {avg_improvement:+.4f}")
        print(f"  Max Improvement: {max_improvement:+.4f}")
        print(f"  Min Improvement: {min_improvement:+.4f}")
        print(f"  Better in {(feature_results['Improvement'] > 0).sum()}/{len(feature_results)} bins")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
