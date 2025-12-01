"""
Calibration Plot for Random Forest Classifier
Assesses if predicted probabilities match observed outcomes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from data_preprocessing import load_and_process_mic_data
from feature_extraction_enhanced import get_features

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTER_TEST_SIZE = 0.2
RANDOM_STATE = 42

filepath = "../data/"
output_path = "../output/"
cache_path = "../cache/"
os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def calculate_ece(y_true, y_pred, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    bin_totals = 0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 80)
    print("RANDOM FOREST CALIBRATION ANALYSIS")
    print("=" * 80)
    
    print("\nPhase 1: Loading and preparing data...")
    
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
    y_train = labels.iloc[train_indices].copy().reset_index(drop=True)
    
    df_test = df.iloc[test_indices].copy().reset_index(drop=True)
    y_test = labels.iloc[test_indices].copy().reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("CLASS BALANCE DETAILS")
    print("=" * 80)
    
    # Training set balance
    train_active = sum(y_train == 1)
    train_inactive = sum(y_train == 0)
    train_total = len(y_train)
    
    print(f"\nTraining Set (n={train_total}):")
    print(f"  Active (1):   {train_active:6d} ({100*train_active/train_total:5.1f}%)")
    print(f"  Inactive (0): {train_inactive:6d} ({100*train_inactive/train_total:5.1f}%)")
    print(f"  Ratio (Active:Inactive): 1:{train_inactive/train_active:.2f}")
    
    # Test set balance
    test_active = sum(y_test == 1)
    test_inactive = sum(y_test == 0)
    test_total = len(y_test)
    
    print(f"\nTest Set (n={test_total}):")
    print(f"  Active (1):   {test_active:6d} ({100*test_active/test_total:5.1f}%)")
    print(f"  Inactive (0): {test_inactive:6d} ({100*test_inactive/test_total:5.1f}%)")
    print(f"  Ratio (Active:Inactive): 1:{test_inactive/test_active:.2f}")
    
    # Overall balance
    overall_active = train_active + test_active
    overall_inactive = train_inactive + test_inactive
    overall_total = len(labels)
    
    print(f"\nOverall (n={overall_total}):")
    print(f"  Active (1):   {overall_active:6d} ({100*overall_active/overall_total:5.1f}%)")
    print(f"  Inactive (0): {overall_inactive:6d} ({100*overall_inactive/overall_total:5.1f}%)")
    print(f"  Ratio (Active:Inactive): 1:{overall_inactive/overall_active:.2f}")
    
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION")
    print("=" * 80)
    
    # Load or compute features
    cache_file_train = os.path.join(cache_path, "X_train_rdkit.parquet")
    if os.path.exists(cache_file_train):
        print("Loading cached training features...")
        X_train = pd.read_parquet(cache_file_train)
    else:
        print("Computing training features...")
        X_train = get_features(df_train, method='rdkit', batch_size=1000)
        X_train.to_parquet(cache_file_train, compression='snappy')
    
    cache_file_test = os.path.join(cache_path, "X_test_rdkit.parquet")
    if os.path.exists(cache_file_test):
        print("Loading cached test features...")
        X_test = pd.read_parquet(cache_file_test)
    else:
        print("Computing test features...")
        X_test = get_features(df_test, method='rdkit', batch_size=1000)
        X_test.to_parquet(cache_file_test, compression='snappy')
    
    print(f"Training features: {X_train.shape}")
    print(f"Test features: {X_test.shape}")
    
    print("\n" + "=" * 80)
    print("PHASE 2: TRAINING MODEL")
    print("=" * 80)
    
    # Train RF model
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=1200,
        max_depth=None,
        max_features='sqrt',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✓ Random Forest model trained")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate calibration metrics
    brier = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    ece = calculate_ece(y_test.values, y_pred_proba)
    
    print(f"\nCalibration Metrics:")
    print(f"  Brier Score: {brier:.4f} (lower is better, 0 = perfect)")
    print(f"  Log Loss:    {logloss:.4f} (lower is better)")
    print(f"  ECE:         {ece:.4f} (lower is better, 0 = perfect calibration)")
    
    print("\n" + "=" * 80)
    print("PHASE 3: CREATING CALIBRATION PLOTS")
    print("=" * 80)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # =========================================================================
    # Plot 1: Calibration Curve (Reliability Diagram)
    # =========================================================================
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax1.plot(prob_pred, prob_true, 'o-', label='RF Classifier', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curve (Reliability Diagram)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Add metrics text
    textstr = f'Brier: {brier:.4f}\nLog Loss: {logloss:.4f}\nECE: {ece:.4f}'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =========================================================================
    # Plot 2: Histogram of Predicted Probabilities
    # =========================================================================
    n_bins_hist = 20
    
    # Positive class
    ax2.hist(y_pred_proba[y_test == 1], bins=n_bins_hist, alpha=0.6, 
             label=f'Active (n={sum(y_test == 1)})', color='green', edgecolor='black')
    
    # Negative class
    ax2.hist(y_pred_proba[y_test == 0], bins=n_bins_hist, alpha=0.6, 
             label=f'Inactive (n={sum(y_test == 0)})', color='red', edgecolor='black')
    
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Predicted Probabilities', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    calibration_plot = os.path.join(output_path, "rf_calibration_plot.png")
    plt.savefig(calibration_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Calibration plot saved: {calibration_plot}")
    plt.close()
    
    # =========================================================================
    # Plot 3: Detailed Calibration Histogram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    observed_freq = []
    predicted_mean = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        if i == n_bins - 1:  # Include right edge in last bin
            in_bin = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i+1])
        
        if in_bin.sum() > 0:
            observed_freq.append(y_test[in_bin].mean())
            predicted_mean.append(y_pred_proba[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            observed_freq.append(0)
            predicted_mean.append(bin_centers[i])
            bin_counts.append(0)
    
    # Create bar chart
    x = np.arange(n_bins)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, predicted_mean, width, label='Mean Predicted Probability', 
                   alpha=0.8, color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, observed_freq, width, label='Observed Frequency', 
                   alpha=0.8, color='orange', edgecolor='black')
    
    # Add count labels on bars
    for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, bin_counts)):
        if count > 0:
            height = max(bar1.get_height(), bar2.get_height())
            ax.text(i, height + 0.02, f'n={count}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Probability Bins', fontsize=12)
    ax.set_ylabel('Probability / Frequency', fontsize=12)
    ax.set_title('Detailed Calibration: Predicted vs Observed by Bin', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(n_bins)], 
                       rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    calibration_hist = os.path.join(output_path, "rf_calibration_histogram.png")
    plt.savefig(calibration_hist, dpi=300, bbox_inches='tight')
    print(f"✓ Calibration histogram saved: {calibration_hist}")
    plt.close()
    
    # =========================================================================
    # Save detailed results
    # =========================================================================
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'bin_lower': bin_edges[:-1],
        'bin_upper': bin_edges[1:],
        'predicted_mean': predicted_mean,
        'observed_freq': observed_freq,
        'count': bin_counts,
        'calibration_error': np.abs(np.array(predicted_mean) - np.array(observed_freq))
    })
    
    results_file = os.path.join(output_path, "rf_calibration_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"✓ Calibration results saved: {results_file}")
    
    # Summary metrics
    summary = pd.DataFrame({
        'Metric': ['Brier Score', 'Log Loss', 'Expected Calibration Error (ECE)', 
                   'Test Set Size', 'Positive Class Count', 'Negative Class Count'],
        'Value': [brier, logloss, ece, len(y_test), sum(y_test == 1), sum(y_test == 0)]
    })
    
    summary_file = os.path.join(output_path, "rf_calibration_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"✓ Summary saved: {summary_file}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nFiles saved to: {output_path}")
    print("  - rf_calibration_plot.png")
    print("  - rf_calibration_histogram.png")
    print("  - rf_calibration_results.csv")
    print("  - rf_calibration_summary.csv")


if __name__ == '__main__':
    main()
