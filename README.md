# CAMPER: Mechanistic AI for Anti-MRSA Peptides

A machine learning system combining Random Forest predictions with biophysical property scoring for antimicrobial peptide discovery. Optimized for **precision over recall** — prioritizes avoiding failed synthesis attempts over exhaustive candidate identification.

## Quick Start

### Installation

```bash
pip install scikit-learn pandas numpy joblib python-Levenshtein RDKit biopython peptides openpyxl matplotlib seaborn
```

### Project Structure

```
src/
├── BiophysicalPeptideScorer.py      # Core scoring class
├── feature_extraction_enhanced.py    # RDKit + biophysical feature extraction
├── data_preprocessing.py              # Data loading and preparation
├── peptide_screening.py           # Main screening script
├── cv_utilities.py                   # Cross-validation helpers
├── matrix_based_split.py             # Homology-aware splitting
└── [other analysis scripts]
```

## Main Workflow: Peptide Screening

### 1. Prepare Training Data

Place your MIC data in Excel format:
```
Columns: sequence, active (0 or 1)
```

### 2. Run Peptide Screening

```bash
python peptide_screening.py
```

**What it does:**

1. **Loads training data** from `../data/staph_MICs_2024_10_14_updated.xlsx`
2. **Extracts features** (RDKit descriptors + biophysical properties: GRAVY, helicity, charge, hydrophobic moment)
3. **Trains Random Forest** model on 80% of data
4. **Generates peptide variants** from base sequence with position-specific substitutions
5. **Scores all variants** using RF + BiophysicalPeptideScorer ensemble
6. **Exports ranked results** to `predictions.xlsx`

### 3. Customize Peptide Variants

Edit these lines in `peptide_screening.py`:

```python
WP_BASE_SEQUENCE = "WKLIAKLAKKLL"      # Your base sequence
WP_POSITIONS = [2, 5, 6, 10]           # 0-based positions to modify
WP_AMINO_ACIDS = "ADCEFGHIKLMNPQRSTVWY"  # Amino acids to test per position
```

**Example output:** Modifying 4 positions with 20 amino acids = 160,000 variants

### 4. Interpret Results

The exported Excel file contains:

| Column | Description |
|--------|-------------|
| sequence | Modified peptide sequence |
| gravy, helicity, charge, hydrophobic_moment | Biophysical properties |
| original_probability | RF model confidence (0-1) |
| biophysical_score | Biophysical favorability (0-1) |
| custom_score | **Final ranking score** (weighted ensemble) |
| charge_score, helicity_effect, gravy_effect, amphipathicity_score | Component contributions |

**Ranking:** Sort by `custom_score` (highest = most promising)

## Configuration

### Model Parameters

Edit `BiophysicalPeptideScorer` initialization in any script:

```python
scorer = BiophysicalPeptideScorer(
    w_prob=0.5,                        # Weight for RF probability
    charge_opt_orig=3.0,               # Optimal charge
    helicity_threshold_orig=0.7,       # Minimum helicity
    amphipathicity_threshold_orig=0.3, # Minimum amphipathicity
    gravy_threshold_orig=0.7,          # Minimum hydrophobicity
    # ... other parameters
)
```

### Feature Extraction

Control batch size for large datasets (reduces memory):

```python
X = get_features(df, method='rdkit', batch_size=5000)  # Default: 1000
```

## Other Analysis Scripts

All scripts save results to `../output/`:

| Script | Purpose                                                         |
|--------|-----------------------------------------------------------------|
| `calibration_plot.py` | Assess model probability calibration                            |
| `compare_models_cv.py` | Compare homology-aware vs random split CV                       |
| `confusion_matrices_holdout.py` | Confusion matrices on holdout test set                          |
| `feature_importance.py` | Top 10/20 most important features                               |
| `feature_space_performance.py` | Where each method excels                                        |
| `precision_improvement.py` | Precision gains across feature bins                             |
| `y_randomization.py` | Validate model isn't overfitting                                |
| `matrix_sensitivity_analysis.py` | Test different similarity thresholds                            |
| `optimized_cv_with_plots.py` | Compare multiple model-feature engineering combinations with CV |

## Key Design Principles

- **Precision over Recall:** Avoids predicting inactive peptides as active (costly synthesis failures)
- **Biophysical Grounding:** Incorporates known antimicrobial peptide properties
- **Homology Awareness:** Prevents sequence similarity bias in cross-validation (Levenshtein distance)
- **Modular Architecture:** Reusable components with intelligent caching for efficiency

## Important Notes

### Data Requirements
- Minimum ~200 unique sequences for reliable training
- Class balance matters (active:inactive ratio affects threshold selection)
- Sequences should be peptides (amino acid letters only)

### Homology-Aware Cross-Validation

By default, the system uses **random splits**. For more conservative train-test splits, enable homology-aware splitting:

```python
from matrix_based_split import SimilarityMatrix, homology_aware_split

# Create similarity matrix once
similarity_matrix = SimilarityMatrix(df['sequence'].values)
train_idx, test_idx = homology_aware_split(
    similarity_matrix, labels,
    test_size=0.2,
    similarity_threshold=0.80
)
```

This prevents similar sequences from appearing in both train and test sets.

### Biological Validation

CAMPER was developed using a validated peptide from biological experiments. The model shows strong performance in specific biophysical regions (high GRAVY + helicity) but varies across feature space. **Validate top candidates experimentally.**

## Caching Strategy

Features and similarity matrices are cached in `../cache/`:
- Dramatically speeds up repeated analyses
- Parquet format for fast I/O
- Delete cache to force recomputation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory error during feature extraction | Reduce `batch_size` in `get_features()` |
| Slow similarity matrix computation | Use parallel jobs: `SimilarityMatrix(sequences, n_jobs=-1)` |
| RDKit warnings | Safe to ignore; handled gracefully in code |
| Missing columns in output | Ensure input Excel has `sequence` and `active` columns |

## Citation

If using CAMPER, cite:
- Shehadeh, F., Mishra, B., Ferrer-Espada, R. et al. CAMPER: mechanistic artificial intelligence for designing peptides that target MRSA persisters. Nat Commun (2026). https://doi.org/10.1038/s41467-026-70348-9

## Contact & Support

For questions about methodology or modifications, refer to the published manuscript, the model card and inline code documentation.
