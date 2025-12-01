"""
Peptide Screening Script
Generates peptide variants, scores them with RF + BiophysicalPeptideScorer, exports results.
"""

import os
import itertools
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from feature_extraction_enhanced import get_features
from data_preprocessing import load_and_process_mic_data
from BiophysicalPeptideScorer import BiophysicalPeptideScorer

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

# WP Configuration
WP_BASE_SEQUENCE = "WKXIAXXAKKXL"
WP_POSITIONS = [2, 5, 6, 10]
WP_AMINO_ACIDS = "ADCEFGHIKLMNPQRSTVWY"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# =============================================================================
# PREDICT AND EXPORT FUNCTION
# =============================================================================

def predict_and_export_simple(X_WP, df_WP, model, optimal_scorer,
                             output_file="optimized_predictions.xlsx",
                             print_top=10,
                             gravy_column='gravy',
                             helicity_column='helicity',
                             charge_column='charge',
                             hydrophobic_moment_column='hydrophobic_moment'):
    """
    Predict using both original model and optimized scorer, then export results to Excel.
    """
    # Ensure X_WP and df_WP are aligned
    if not X_WP.index.equals(df_WP.index):
        print("Warning: X_WP and df_WP indices don't match. Aligning...")
        if len(X_WP) == len(df_WP):
            X_WP = X_WP.reset_index(drop=True)
            df_WP = df_WP.reset_index(drop=True)
            print("Aligned using reset_index()")
        else:
            raise ValueError("X_WP and df_WP have different lengths. Cannot proceed.")

    # Get original model predictions
    original_probs = model.predict_proba(X_WP)[:, 1]

    # Get optimized scorer predictions
    _, scores_df = optimal_scorer.predict(X_WP, model)
    custom_scores = scores_df['final_score'].values
    biophysical_scores = scores_df['biophysical_score'].values
    charge_scores = scores_df['charge_score'].values
    helicity_scores = scores_df['helicity_effect'].values
    gravy_scores = scores_df['gravy_effect'].values
    ampipathy_scores = scores_df['amphipathicity_score'].values
    cooperative_effects = scores_df['cooperative_effect'].values
    cooperative_boost = scores_df['cooperative_boost'].values

    # Create results DataFrame
    results = pd.DataFrame()

    # Add sequence information first
    results['sequence'] = df_WP['sequence'].values

    # Add property columns next
    for col in [gravy_column, helicity_column, charge_column, hydrophobic_moment_column]:
        if col in X_WP.columns:
            results[col] = X_WP[col].values

    # Add predictions and scores
    results['original_probability'] = original_probs
    results['biophysical_score'] = biophysical_scores
    results['charge_score'] = charge_scores
    results['helicity_effect'] = helicity_scores
    results['gravy_effect'] = gravy_scores
    results['amphipathicity_score'] = ampipathy_scores
    results['cooperative_effect'] = cooperative_effects
    results['cooperative_boost'] = cooperative_boost
    results['custom_score'] = custom_scores

    # Sort by custom score (descending)
    results_sorted = results.sort_values('custom_score', ascending=False)

    # Print top peptides
    print(f"\nTop {print_top} peptides by custom score:")
    top_results = results_sorted.head(print_top)
    print(top_results.to_string(index=False))

    # Save to Excel
    try:
        results_sorted.to_excel(output_file, index=False)
        print(f"\n✓ Results saved to '{output_file}'")
    except Exception as e:
        print(f"\nError saving Excel file: {e}")
        results_sorted.to_csv(output_file.replace('.xlsx', '.csv'), index=False)
        print(f"Saved results to CSV instead")

    return results_sorted

def main():
    print("=" * 80)
    print("PHASE 1: LOADING AND PREPARING DATA")
    print("=" * 80)
    
    # Load data
    MICfile = os.path.join(filepath, "staph_MICs_2024_10_14_updated.xlsx")
    df, labels = load_and_process_mic_data(MICfile)
    
    # Split into train/test
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=OUTER_TEST_SIZE,
        random_state=RANDOM_STATE
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
        X_train.to_parquet(cache_file_train, compression='snappy')
    
    print(f"Training features shape: {X_train.shape}")
    
    print("\n" + "=" * 80)
    print("PHASE 2: TRAINING RANDOM FOREST MODEL")
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
    
    print("\n" + "=" * 80)
    print("PHASE 3: CREATING BIOPHYSICAL SCORER")
    print("=" * 80)
    
    scorer = BiophysicalPeptideScorer(
        gravy_column='gravy',
        helicity_column='helicity',
        charge_column='charge',
        hydrophobic_moment_column='hydrophobic_moment',
        normalize_properties=False
    )
    scorer.fit(X_train)
    print("✓ BiophysicalPeptideScorer created and fitted")
    
    print("\n" + "=" * 80)
    print("PHASE 4: GENERATING WP VARIANTS")
    print("=" * 80)
    
    peptide = list(WP_BASE_SEQUENCE)
    print(f"Base sequence: {''.join(peptide)}")
    print(f"Positions to modify: {WP_POSITIONS}")
    print(f"Amino acids: {WP_AMINO_ACIDS} ({len(WP_AMINO_ACIDS)} options per position)")
    
    # Generate variants
    possible_substitutions = [WP_AMINO_ACIDS] * len(WP_POSITIONS)
    combinations = itertools.product(*possible_substitutions)
    
    modified_peptides = []
    for combo in combinations:
        new_peptide = peptide.copy()
        for idx, pos in enumerate(WP_POSITIONS):
            new_peptide[pos] = combo[idx]
        modified_peptides.append("".join(new_peptide))
    
    n_variants = len(modified_peptides)
    print(f"✓ Generated {n_variants} variants")
    print(f"  Formula: {len(WP_AMINO_ACIDS)}^{len(WP_POSITIONS)} = {n_variants}")
    
    df_WP = pd.DataFrame(modified_peptides, columns=['sequence'])
    
    print("\n" + "=" * 80)
    print("PHASE 5: EXTRACTING FEATURES FOR WP VARIANTS")
    print("=" * 80)
    
    cache_file_wp = os.path.join(cache_path, "X_WP_rdkit.parquet")
    if os.path.exists(cache_file_wp):
        print("Loading cached WP features...")
        X_WP = pd.read_parquet(cache_file_wp)
    else:
        print("Computing WP features...")
        X_WP = get_features(df_WP, method='rdkit', batch_size=1000)
        X_WP.to_parquet(cache_file_wp, compression='snappy')
    
    print(f"WP features shape: {X_WP.shape}")
    
    print("\n" + "=" * 80)
    print("PHASE 6: SCORING AND EXPORTING")
    print("=" * 80)
    
    output_file = os.path.join(output_path, "predictions.xlsx")
    
    results = predict_and_export_simple(
        X_WP=X_WP,
        df_WP=df_WP,
        model=rf_model,
        optimal_scorer=scorer,
        output_file=output_file,
        print_top=20
    )
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
