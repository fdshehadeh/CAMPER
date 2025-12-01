"""
Y-Randomization Test
Validates model performance by comparing against models trained on shuffled labels.
"""

import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_preprocessing import load_and_process_mic_data

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTER_TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ITERATIONS = 100

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
    print("Y-RANDOMIZATION TEST")
    print("=" * 80)
    
    print("\nPhase 1: Loading data...")
    
    # Load data
    MICfile = os.path.join(filepath, "staph_MICs_2024_10_14_updated.xlsx")
    df, labels = load_and_process_mic_data(MICfile)
    
    print(f"Loaded {len(df)} peptides")
    
    # Split into train/test
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=OUTER_TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    y_train = labels.iloc[train_indices].copy().reset_index(drop=True)
    y_test = labels.iloc[test_indices].copy().reset_index(drop=True)
    
    # Load features
    cache_file_train = os.path.join(cache_path, "X_train_rdkit.parquet")
    if os.path.exists(cache_file_train):
        print("Loading cached training features...")
        X_train = pd.read_parquet(cache_file_train)
    else:
        from feature_extraction_enhanced import get_features
        print("Computing training features...")
        df_train = df.iloc[train_indices].copy().reset_index(drop=True)
        X_train = get_features(df_train, method='rdkit', batch_size=1000)
        X_train.to_parquet(cache_file_train, compression='snappy')
    
    cache_file_test = os.path.join(cache_path, "X_test_rdkit.parquet")
    if os.path.exists(cache_file_test):
        print("Loading cached test features...")
        X_test = pd.read_parquet(cache_file_test)
    else:
        from feature_extraction_enhanced import get_features
        print("Computing test features...")
        df_test = df.iloc[test_indices].copy().reset_index(drop=True)
        X_test = get_features(df_test, method='rdkit', batch_size=1000)
        X_test.to_parquet(cache_file_test, compression='snappy')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    print("\n" + "=" * 80)
    print("TRAINING ORIGINAL MODEL")
    print("=" * 80)
    
    # Train model on original labels
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=1200,
        max_depth=None,
        max_features='sqrt',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics for original model
    original_accuracy = accuracy_score(y_test, y_pred)
    original_precision = precision_score(y_test, y_pred)
    original_recall = recall_score(y_test, y_pred)
    original_f1 = f1_score(y_test, y_pred)
    
    print(f"\nOriginal Model Performance:")
    print(f"  Accuracy:  {original_accuracy:.4f}")
    print(f"  Precision: {original_precision:.4f}")
    print(f"  Recall:    {original_recall:.4f}")
    print(f"  F1 Score:  {original_f1:.4f}")
    
    print("\n" + "=" * 80)
    print(f"Y-RANDOMIZATION ({N_ITERATIONS} iterations)")
    print("=" * 80)
    
    randomized_accuracies = []
    randomized_precisions = []
    randomized_recalls = []
    randomized_f1s = []
    
    for i in range(N_ITERATIONS):
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i + 1}/{N_ITERATIONS}")
        
        # Shuffle labels
        y_train_shuffled = np.random.permutation(y_train)
        
        # Train model on shuffled labels
        model.fit(X_train, y_train_shuffled)
        y_pred_random = model.predict(X_test)
        
        # Calculate metrics
        randomized_accuracies.append(accuracy_score(y_test, y_pred_random))
        randomized_precisions.append(precision_score(y_test, y_pred_random))
        randomized_recalls.append(recall_score(y_test, y_pred_random))
        randomized_f1s.append(f1_score(y_test, y_pred_random))
    
    # Calculate mean and std for randomized models
    mean_accuracy = np.mean(randomized_accuracies)
    std_accuracy = np.std(randomized_accuracies)
    
    mean_precision = np.mean(randomized_precisions)
    std_precision = np.std(randomized_precisions)
    
    mean_recall = np.mean(randomized_recalls)
    std_recall = np.std(randomized_recalls)
    
    mean_f1 = np.mean(randomized_f1s)
    std_f1 = np.std(randomized_f1s)
    
    print(f"\nRandomized Model Performance (mean ± std):")
    print(f"  Accuracy:  {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"  Recall:    {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"  F1 Score:  {mean_f1:.4f} ± {std_f1:.4f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON: ORIGINAL vs RANDOMIZED")
    print("=" * 80)
    
    print(f"\nAccuracy Difference:  {original_accuracy - mean_accuracy:.4f}")
    print(f"  Original:   {original_accuracy:.4f}")
    print(f"  Randomized: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    print(f"\nPrecision Difference: {original_precision - mean_precision:.4f}")
    print(f"  Original:   {original_precision:.4f}")
    print(f"  Randomized: {mean_precision:.4f} ± {std_precision:.4f}")
    
    print(f"\nRecall Difference:    {original_recall - mean_recall:.4f}")
    print(f"  Original:   {original_recall:.4f}")
    print(f"  Randomized: {mean_recall:.4f} ± {std_recall:.4f}")
    
    print(f"\nF1 Difference:        {original_f1 - mean_f1:.4f}")
    print(f"  Original:   {original_f1:.4f}")
    print(f"  Randomized: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Save results to file
    results_df = pd.DataFrame({
        'iteration': range(N_ITERATIONS),
        'accuracy': randomized_accuracies,
        'precision': randomized_precisions,
        'recall': randomized_recalls,
        'f1': randomized_f1s
    })
    
    results_file = os.path.join(output_path, "y_randomization_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Detailed results saved to: {results_file}")
    
    # Create comparison summary
    summary_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
        'Original': [original_accuracy, original_precision, original_recall, original_f1],
        'Randomized_Mean': [mean_accuracy, mean_precision, mean_recall, mean_f1],
        'Randomized_Std': [std_accuracy, std_precision, std_recall, std_f1],
        'Difference': [
            original_accuracy - mean_accuracy,
            original_precision - mean_precision,
            original_recall - mean_recall,
            original_f1 - mean_f1
        ]
    })
    
    summary_file = os.path.join(output_path, "y_randomization_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
