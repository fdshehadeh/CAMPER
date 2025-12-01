"""
Script to generate feature importance plots for the Random Forest model.
Trains RF on the training set and visualizes feature importances.
Creates two plots: top 10 and top 20 features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Import custom modules
from data_preprocessing import load_and_process_mic_data
from feature_extraction_enhanced import get_features

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
    y_train = labels.iloc[train_indices].copy().reset_index(drop=True)
    
    print(f"Training set size: {len(df_train)}")
    
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
    
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 80)
    
    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=1200,
        max_depth=None,
        max_features='sqrt',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("✓ Random Forest model trained")
    
    # =========================================================================
    # EXTRACT FEATURE IMPORTANCES
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING FEATURE IMPORTANCES")
    print("=" * 80)
    
    feature_importance = rf_model.feature_importances_
    feature_names = X_train.columns
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTotal features: {len(importance_df)}")
    print(f"\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save feature importances to CSV
    importance_df.to_csv(os.path.join(output_path, 'feature_importances.csv'), index=False)
    print("\n✓ Saved: feature_importances.csv")
    
    # =========================================================================
    # PLOT TOP 10 FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING TOP 10 FEATURE IMPORTANCE PLOT")
    print("=" * 80)
    
    top_10 = importance_df.head(10).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(range(len(top_10)), top_10['Importance'].values, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_10.iterrows()):
        ax.text(row['Importance'] + 0.001, i, f"{row['Importance']:.4f}", 
                va='center', fontsize=10, weight='bold')
    
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10['Feature'].values, fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
    ax.set_title('Top 10 Most Important Features - Random Forest Model', 
                 fontsize=14, weight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'feature_importance_top10.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'feature_importance_top10.pdf'), 
                bbox_inches='tight')
    print("✓ Saved: feature_importance_top10.png / pdf")
    plt.close()
    
    # =========================================================================
    # PLOT TOP 20 FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING TOP 20 FEATURE IMPORTANCE PLOT")
    print("=" * 80)
    
    top_20 = importance_df.head(20).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    bars = ax.barh(range(len(top_20)), top_20['Importance'].values, color='seagreen', alpha=0.8)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax.text(row['Importance'] + 0.0005, i, f"{row['Importance']:.4f}", 
                va='center', fontsize=9, weight='bold')
    
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['Feature'].values, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
    ax.set_title('Top 20 Most Important Features - Random Forest Model', 
                 fontsize=14, weight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'feature_importance_top20.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'feature_importance_top20.pdf'), 
                bbox_inches='tight')
    print("✓ Saved: feature_importance_top20.png / pdf")
    plt.close()
    
    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 80)
    
    top_10_sum = top_10['Importance'].sum()
    top_20_sum = top_20['Importance'].sum()
    total_importance = importance_df['Importance'].sum()
    
    print(f"\nTotal feature importance (sum): {total_importance:.4f}")
    print(f"Top 10 features account for: {top_10_sum:.4f} ({top_10_sum/total_importance*100:.2f}%)")
    print(f"Top 20 features account for: {top_20_sum:.4f} ({top_20_sum/total_importance*100:.2f}%)")
    
    # Identify biophysical features
    biophys_features = ['gravy', 'helicity', 'charge', 'hydrophobic_moment']
    biophys_importance = importance_df[importance_df['Feature'].isin(biophys_features)]
    
    print(f"\nBiophysical Features Importance:")
    print(biophys_importance.to_string(index=False))
    
    biophys_in_top10 = biophys_importance[biophys_importance.index.isin(top_10.index)]
    biophys_in_top20 = biophys_importance[biophys_importance.index.isin(top_20.index)]
    
    print(f"\nBiophysical features in top 10: {len(biophys_in_top10)}/4")
    if len(biophys_in_top10) > 0:
        print(biophys_in_top10.to_string(index=False))
    
    print(f"\nBiophysical features in top 20: {len(biophys_in_top20)}/4")
    if len(biophys_in_top20) > 0:
        print(biophys_in_top20.to_string(index=False))
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
