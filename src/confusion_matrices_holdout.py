"""
Script to generate confusion matrices for RF and CAMPER on the holdout test set.
Trains models on the training set and evaluates on the holdout test set.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)

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
    print("TRAINING MODELS ON HOLDOUT TRAINING SET")
    print("=" * 80)
    
    # =========================================================================
    # Train Random Forest
    # =========================================================================
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
    
    # =========================================================================
    # Train BiophysicalPeptideScorer (CAMPER)
    # =========================================================================
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
    print("GENERATING PREDICTIONS ON HOLDOUT TEST SET")
    print("=" * 80)
    
    # =========================================================================
    # RF Predictions
    # =========================================================================
    print("\nGenerating RF predictions...")
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    print(f"✓ RF predictions generated")
    
    # =========================================================================
    # CAMPER Predictions
    # =========================================================================
    print("\nGenerating CAMPER predictions...")
    _, camper_scores = scorer.predict(X_test, rf_model)
    camper_proba = camper_scores['final_score'].values
    camper_pred = (camper_proba >= 0.5).astype(int)
    print(f"✓ CAMPER predictions generated")
    
    # =========================================================================
    # GENERATE CONFUSION MATRICES
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING CONFUSION MATRIX PLOTS")
    print("=" * 80)
    
    # Calculate confusion matrices
    cm_rf = confusion_matrix(y_test, rf_pred)
    cm_camper = confusion_matrix(y_test, camper_pred)
    
    # Create side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RF Confusion Matrix
    sns.heatmap(
        cm_rf,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0],
        cbar=False,
        square=True,
        annot_kws={'size': 14, 'weight': 'bold'},
        xticklabels=['Inactive', 'Active'],
        yticklabels=['Inactive', 'Active']
    )
    axes[0].set_title('Random Forest', fontsize=14, weight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, weight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, weight='bold')
    
    # CAMPER Confusion Matrix
    sns.heatmap(
        cm_camper,
        annot=True,
        fmt='d',
        cmap='Greens',
        ax=axes[1],
        cbar=False,
        square=True,
        annot_kws={'size': 14, 'weight': 'bold'},
        xticklabels=['Inactive', 'Active'],
        yticklabels=['Inactive', 'Active']
    )
    axes[1].set_title('CAMPER', fontsize=14, weight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, weight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, weight='bold')
    
    plt.suptitle('Confusion Matrices: Holdout Test Set', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, 'confusion_matrices_holdout.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'confusion_matrices_holdout.pdf'), 
                bbox_inches='tight')
    print("\nSaved: confusion_matrices_holdout.png / pdf")
    plt.close()
    
    # =========================================================================
    # CALCULATE PERFORMANCE METRICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS - HOLDOUT TEST SET")
    print("=" * 80)
    
    metrics_data = []
    
    # RF Metrics
    print("\nRandom Forest:")
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
    
    print(f"  Accuracy:  {rf_acc:.4f}")
    print(f"  Precision: {rf_prec:.4f}")
    print(f"  Recall:    {rf_rec:.4f}")
    print(f"  F1 Score:  {rf_f1:.4f}")
    
    metrics_data.append({
        'Model': 'Random Forest',
        'Accuracy': rf_acc,
        'Precision': rf_prec,
        'Recall': rf_rec,
        'F1 Score': rf_f1
    })
    
    # CAMPER Metrics
    print("\nBiophysicalPeptideScorer (CAMPER):")
    camper_acc = accuracy_score(y_test, camper_pred)
    camper_prec = precision_score(y_test, camper_pred, zero_division=0)
    camper_rec = recall_score(y_test, camper_pred, zero_division=0)
    camper_f1 = f1_score(y_test, camper_pred, zero_division=0)
    
    print(f"  Accuracy:  {camper_acc:.4f}")
    print(f"  Precision: {camper_prec:.4f}")
    print(f"  Recall:    {camper_rec:.4f}")
    print(f"  F1 Score:  {camper_f1:.4f}")
    
    metrics_data.append({
        'Model': 'CAMPER',
        'Accuracy': camper_acc,
        'Precision': camper_prec,
        'Recall': camper_rec,
        'F1 Score': camper_f1
    })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_path, 'holdout_metrics.csv'), index=False)
    print("\nSaved: holdout_metrics.csv")
    
    print("\n" + "=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)
    
    # =========================================================================
    # CONFUSION MATRIX DETAILS
    # =========================================================================
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX DETAILS")
    print("=" * 80)
    
    print("\nRandom Forest Confusion Matrix:")
    print(f"  True Negatives:  {cm_rf[0, 0]}")
    print(f"  False Positives: {cm_rf[0, 1]}")
    print(f"  False Negatives: {cm_rf[1, 0]}")
    print(f"  True Positives:  {cm_rf[1, 1]}")
    
    print("\n(CAMPER) Confusion Matrix:")
    print(f"  True Negatives:  {cm_camper[0, 0]}")
    print(f"  False Positives: {cm_camper[0, 1]}")
    print(f"  False Negatives: {cm_camper[1, 0]}")
    print(f"  True Positives:  {cm_camper[1, 1]}")
    
    # =========================================================================
    # VIOLIN PLOTS: BIOPHYSICAL FEATURES BY PREDICTION OUTCOME
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VIOLIN PLOTS FOR BIOPHYSICAL FEATURES")
    print("=" * 80)
    
    # Identify prediction outcome groups for RF
    rf_tp = (rf_pred == 1) & (y_test == 1)
    rf_fn = (rf_pred == 0) & (y_test == 1)
    
    # Identify prediction outcome groups for CAMPER
    camper_tp = (camper_pred == 1) & (y_test == 1)
    camper_fn = (camper_pred == 0) & (y_test == 1)
    
    print(f"\nRF True Positives: {np.sum(rf_tp)}")
    print(f"RF False Negatives: {np.sum(rf_fn)}")
    print(f"CAMPER True Positives: {np.sum(camper_tp)}")
    print(f"CAMPER False Negatives: {np.sum(camper_fn)}")
    
    # Create dataframe for violin plots
    biophys_features = ['gravy', 'helicity', 'charge', 'hydrophobic_moment']
    plot_data = []
    
    for feature in biophys_features:
        feature_vals = X_test[feature].values
        
        # RF TP
        plot_data.extend([
            {'Feature': feature, 'Value': val, 'Group': 'RF TP'}
            for val in feature_vals[rf_tp]
        ])
        
        # RF FN
        plot_data.extend([
            {'Feature': feature, 'Value': val, 'Group': 'RF FN'}
            for val in feature_vals[rf_fn]
        ])
        
        # CAMPER TP
        plot_data.extend([
            {'Feature': feature, 'Value': val, 'Group': 'CAMPER TP'}
            for val in feature_vals[camper_tp]
        ])
        
        # CAMPER FN
        plot_data.extend([
            {'Feature': feature, 'Value': val, 'Group': 'CAMPER FN'}
            for val in feature_vals[camper_fn]
        ])
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create 2x2 subplot for each biophysical feature
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define color palette: RF vs CAMPER with TP darker/more saturated than FN
    palette = {
        'RF TP': '#4C72B0',        # Darker blue for RF TP
        'CAMPER TP': '#55A868',    # Darker green for CAMPER TP
        'RF FN': '#B3CDE3',        # Lighter blue for RF FN
        'CAMPER FN': '#C5E1A5'     # Lighter green for CAMPER FN
    }
    
    # Define order: TP groups together, then FN groups
    group_order = ['RF TP', 'CAMPER TP', 'RF FN', 'CAMPER FN']
    
    for idx, feature in enumerate(biophys_features):
        ax = axes[idx]
        feature_data = plot_df[plot_df['Feature'] == feature]
        
        sns.boxplot(
            data=feature_data,
            x='Group',
            y='Value',
            ax=ax,
            palette=palette,
            order=group_order,
            width=0.6,
            linewidth=1.5
        )
        
        # Add individual data points
        sns.stripplot(
            data=feature_data,
            x='Group',
            y='Value',
            ax=ax,
            color='black',
            alpha=0.4,
            size=4,
            jitter=True,
            order=group_order
        )
        
        ax.set_title(f'{feature.capitalize()}', fontsize=12, weight='bold')
        ax.set_xlabel('Group', fontsize=11, weight='bold')
        ax.set_ylabel('Value', fontsize=11, weight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Biophysical Features: True Positives vs False Negatives\nRandom Forest vs CAMPER', 
                 fontsize=14, weight='bold', y=1.00)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_path, 'biophysical_features_boxplot.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'biophysical_features_boxplot.pdf'), 
                bbox_inches='tight')
    print("\n✓ Saved: biophysical_features_boxplot.png / pdf")
    plt.close()
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
