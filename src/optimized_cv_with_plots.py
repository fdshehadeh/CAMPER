#%% md
# # Optimized Model & Feature Selection with Comparative Plots
# 
# This script implements a robust and computationally efficient workflow to compare different combinations of feature engineering methods and machine learning models.
# 
# ### Workflow:
# 1. **Outer Homology-Aware Split:** Create a final holdout test set.
# 2. **Feature Pre-computation:** Efficiently generate and cache all feature sets (RDKit, ProtBERT, N-grams) for the training data once.
# 3. **Nested Search & Prediction Collection:** Loop through each feature set and model type, using `RandomizedSearchCV` with a homology-aware splitter to find the best hyperparameters. Collect the out-of-sample predictions for each best-tuned combination.
# 4. **Visual & Tabular Analysis:** Display a summary table of the best scores and generate comparative ROC, PR, and Enrichment plots for all combinations.
# 5. **Final  Evaluation:** Train the single best combination on the full training set and evaluate it once on the holdout test set.
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split


from matplotlib.backends.backend_pdf import PdfPages
from sklearn.utils import resample
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Import custom modules
from data_preprocessing import load_and_process_mic_data
from feature_extraction_enhanced import get_features
from matrix_based_split import SimilarityMatrix
#%% md
# ## 1. Configuration
#%%
SIMILARITY_THRESHOLD = 0.80
OUTER_TEST_SIZE = 0.2
RANDOM_STATE = 42
K_FOLDS =5
N_ITER_SEARCH = 20 
N_BOOTSTRAPS = 100 

# File Paths
filepath = "../data/"
output_path = "../output/"
cache_path = "../cache/"
os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)
#%% md
# ## 2. Data Loading and Outer Split
#%%
MICfile = os.path.join(filepath, "staph_MICs_2024_10_14_updated.xlsx")
df, labels = load_and_process_mic_data(MICfile)

matrix_cache_file = os.path.join(cache_path, "similarity_matrix_staph.npz")
if os.path.exists(matrix_cache_file):
    similarity_matrix = SimilarityMatrix.load(matrix_cache_file)
else:
    similarity_matrix = SimilarityMatrix(df['sequence'].values)
    similarity_matrix.save(matrix_cache_file)

# Create an array of indices representing each row in your dataframe
indices = np.arange(len(df))

# Use train_test_split to get a stratified random split of the *indices*
train_indices, test_indices = train_test_split(
    indices,
    test_size=OUTER_TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels  # Stratify using the full labels Series to maintain class balance
)

df_train = df.iloc[train_indices].copy().reset_index()
df_test = df.iloc[test_indices].copy().reset_index(drop=True)
y_train = labels.iloc[train_indices].copy().reset_index(drop=True)
y_test = labels.iloc[test_indices].copy().reset_index(drop=True)


print(f"Main training set size: {len(df_train)}")
print(f"Final holdout test set size: {len(df_test)}")
#%%
df_train.head()
#%% md
# ## 3. Pre-compute All Feature Sets
#%%
feature_configs = [
    {'name': 'RDKit & Biopython', 'key': 'rdkit', 'params': {'batch_size': 1000}},
    {'name': 'ProtBERT', 'key': 'protbert', 'params': {'batch_size': 32}},
    {'name': 'N-grams', 'key': 'ngrams', 'params': {'min_freq': 2}}
]

X_train_sets = {}

print("--- Pre-computing all feature sets for the training data ---")
for config in feature_configs:
    key, name = config['key'], config['name']
    cache_file = os.path.join(cache_path, f"X_train_{key}.parquet")
    print(f"\nProcessing: {name}...")
    
    if os.path.exists(cache_file):
        X_train_sets[key] = pd.read_parquet(cache_file)
    else:
        features = get_features(df_train, method=key, **config.get('params', {}))
        features.to_parquet(cache_file)
        X_train_sets[key] = features
        
    print(f"Feature set '{name}' ready with shape: {X_train_sets[key].shape}")
#%% md
# ## 4. Define Models, Parameter Grids, and CV Splitter
#%%
class HomologyAwareCV:
    def __init__(self, similarity_matrix, original_indices, threshold, n_splits=5, random_state=42):
        self.n_splits=n_splits; self.random_state=random_state; self.similarity_matrix=similarity_matrix
        self.original_indices=original_indices; self.threshold=threshold
    def get_n_splits(self, X=None, y=None, groups=None): return self.n_splits
    def split(self, X, y=None, groups=None):
        adj_matrix = self.similarity_matrix.matrix[np.ix_(self.original_indices, self.original_indices)] >= self.threshold
        n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        components = [np.where(labels == i)[0] for i in range(n_components)]
        fold_assignments = np.arange(len(components)) % self.n_splits
        np.random.RandomState(self.random_state).shuffle(fold_assignments)
        for i in range(self.n_splits):
            val_idx = np.concatenate([components[j] for j in np.where(fold_assignments == i)[0]])
            train_idx = np.concatenate([components[j] for j in np.where(fold_assignments != i)[0]])
            if len(val_idx) > 0 and len(train_idx) > 0: yield train_idx, val_idx

models_to_test = [
    {'name': 'RandomForest', 'estimator': RandomForestClassifier(random_state=RANDOM_STATE), 'params': {'n_estimators': [1000, 1200, 1600], 'max_depth': [10, 30, None], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']}},
    {'name': 'XGBoost', 'estimator': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'), 'params': {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}},
    {'name': 'LogisticRegression', 'estimator': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 'params': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}}
]
#%% md
# ## 5. Run Search Loop and Collect Predictions
#%%
search_results = []
best_estimators = {}
pipeline_cv_predictions = {}

homology_cv_splitter = HomologyAwareCV(
    similarity_matrix=similarity_matrix, original_indices=df_train['index'].to_numpy(),
    threshold=SIMILARITY_THRESHOLD, n_splits=K_FOLDS, random_state=RANDOM_STATE
)

for ft_config in feature_configs:
    ft_key = ft_config['key']
    ft_name = ft_config['name']
    X_train_current = X_train_sets[ft_key]
    
    for model_config in models_to_test:
        model_name = model_config['name']
        pipeline_name = f"{ft_name} + {model_name}"
        
        print(f"\n{'='*50}\nRUNNING SEARCH FOR: {pipeline_name}\n{'='*50}")
        
        search = RandomizedSearchCV(
            estimator=model_config['estimator'], param_distributions=model_config['params'],
            n_iter=N_ITER_SEARCH, cv=homology_cv_splitter, scoring='roc_auc',
            n_jobs=-1, verbose=1, random_state=RANDOM_STATE, refit=True
        )
        
        search.fit(X_train_current, y_train)
        
        print(f"\nGetting out-of-sample predictions for {pipeline_name}...")
        y_pred_proba_cv = cross_val_predict(
            search.best_estimator_, X_train_current, y_train, 
            cv=homology_cv_splitter, method='predict_proba', n_jobs=-1
        )[:, 1]
        
        pipeline_cv_predictions[pipeline_name] = {'y_true': y_train.to_numpy(), 'y_pred_proba': y_pred_proba_cv}
        
        search_results.append({
            'Pipeline': pipeline_name, 'Feature Method': ft_name, 'Model': model_name,
            'Best CV Score (AUC)': search.best_score_, 'Best Params': search.best_params_
        })
        best_estimators[pipeline_name] = search.best_estimator_
#%% md
# ## 6. Analyze CV Results (Tabular and Visual)
#%%
def plot_combined_pipeline_curves(cv_prediction_data, output_dir='.', n_bootstraps=1000):
    """Generates combined ROC, PR, and Enrichment plots for all pipelines."""
    print("\n--- Generating Combined CV Performance Plots with 95% CIs ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Cross-Validated Performance Comparison of Pipelines', fontsize=18, weight='bold')

    # ROC Plot
    ax_roc = axes[0, 0]
    for name, data in cv_prediction_data.items():
        y_true, y_proba = data['y_true'], data['y_pred_proba']
        boot_aucs = [roc_auc_score(y_true[idx], y_proba[idx]) for idx in (np.random.randint(0, len(y_proba), len(y_proba)) for _ in range(n_bootstraps)) if len(np.unique(y_true[idx])) > 1]
        mean_auc, ci_auc = np.mean(boot_aucs), np.percentile(boot_aucs, [2.5, 97.5])
        RocCurveDisplay.from_predictions(y_true, y_proba, name=f'{name}\nAUROC={mean_auc:.3f} (CI {ci_auc[0]:.3f}-{ci_auc[1]:.3f})', ax=ax_roc)
    ax_roc.set_title('ROC Curve Comparison', fontsize=14, weight='bold')

    # PR Plot
    ax_pr = axes[0, 1]
    for name, data in cv_prediction_data.items():
        y_true, y_proba = data['y_true'], data['y_pred_proba']
        boot_aps = [average_precision_score(y_true[idx], y_proba[idx]) for idx in (np.random.randint(0, len(y_proba), len(y_proba)) for _ in range(n_bootstraps)) if len(np.unique(y_true[idx])) > 1]
        mean_ap, ci_ap = np.mean(boot_aps), np.percentile(boot_aps, [2.5, 97.5])
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=f'{name}\nAUPRC={mean_ap:.3f} (CI {ci_ap[0]:.3f}-{ci_ap[1]:.3f})', ax=ax_pr)
    ax_pr.set_title('PR Curve Comparison', fontsize=14, weight='bold')

    # Full Enrichment Plot
    ax_enr = axes[1, 0]
    for name, data in cv_prediction_data.items():
        df_enrich = pd.DataFrame({'y_true': data['y_true'], 'y_proba': data['y_pred_proba']}).sort_values('y_proba', ascending=False)
        y_enrich = df_enrich['y_true'].cumsum() / df_enrich['y_true'].sum() * 100
        x_enrich = np.arange(1, len(df_enrich) + 1) / len(df_enrich) * 100
        ax_enr.plot(x_enrich, y_enrich, label=name, lw=2)
    ax_enr.plot([0, 100], [0, 100], 'k--', label='Random'); ax_enr.set_title('Enrichment Plot (Full)', fontsize=14, weight='bold')
    ax_enr.set_xlabel('% Library Screened'); ax_enr.set_ylabel('% Actives Found'); ax_enr.legend()

    # Early Enrichment Plot
    ax_zoom = axes[1, 1]
    for name, data in cv_prediction_data.items():
        df_enrich = pd.DataFrame({'y_true': data['y_true'], 'y_proba': data['y_pred_proba']}).sort_values('y_proba', ascending=False)
        y_enrich = df_enrich['y_true'].cumsum() / df_enrich['y_true'].sum() * 100
        x_enrich = np.arange(1, len(df_enrich) + 1) / len(df_enrich) * 100
        ax_zoom.plot(x_enrich, y_enrich, label=name, lw=2)
    ax_zoom.plot([0, 20], [0, 20], 'k--', label='Random'); ax_zoom.set_title('Early Enrichment (Top 20%)', fontsize=14, weight='bold')
    ax_zoom.set_xlabel('% Library Screened'); ax_zoom.set_ylabel('% Actives Found'); ax_zoom.set_xlim(0, 20); ax_zoom.legend()


    plt.tight_layout(rect=[0, 0.03, 1, 0.96]);
    pdf_path = os.path.join(output_dir, 'pipeline_cv_curves_comparison.pdf'); plt.savefig(pdf_path, dpi=300)
    plt.show()
    print(f"Combined CV plots saved to {pdf_path}")

results_df = pd.DataFrame(search_results).sort_values('Best CV Score (AUC)', ascending=False)
print("--- Combined Search Results Summary ---")
display(results_df)

plot_combined_pipeline_curves(pipeline_cv_predictions, output_dir=output_path, n_bootstraps=N_BOOTSTRAPS)

best_pipeline_info = results_df.iloc[0]
best_params = best_pipeline_info['Best Params'] # This defines the variable for the next cell
best_pipeline_name = results_df.iloc[0]['Pipeline']
final_pipeline = best_estimators[best_pipeline_name]
print(f"\n🏆 Best overall pipeline from CV: {best_pipeline_name}")
#%%
def plot_combined_pipeline_curves(cv_prediction_data, output_dir='.', n_bootstraps=1000):
    """
    Generates combined ROC, PR, Enrichment plots, and Confusion Matrices
    for all pipelines.
    """
    print("\n--- Generating Combined CV Performance Plots with 95% CIs ---")

    # --- Part 1: Original Curve Plots (2x2 Grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Cross-Validated Performance Comparison of Pipelines', fontsize=18, weight='bold')

    # ROC Plot
    ax_roc = axes[0, 0]
    for name, data in cv_prediction_data.items():
        y_true, y_proba = data['y_true'], data['y_pred_proba']
        boot_aucs = [roc_auc_score(y_true[idx], y_proba[idx]) for idx in (np.random.randint(0, len(y_proba), len(y_proba)) for _ in range(n_bootstraps)) if len(np.unique(y_true[idx])) > 1]
        mean_auc, ci_auc = np.mean(boot_aucs), np.percentile(boot_aucs, [2.5, 97.5])
        RocCurveDisplay.from_predictions(y_true, y_proba, name=f'{name}\nAUROC={mean_auc:.3f} (CI {ci_auc[0]:.3f}-{ci_auc[1]:.3f})', ax=ax_roc)
    ax_roc.set_title('ROC Curve Comparison', fontsize=14, weight='bold')

    # PR Plot
    ax_pr = axes[0, 1]
    for name, data in cv_prediction_data.items():
        y_true, y_proba = data['y_true'], data['y_pred_proba']
        boot_aps = [average_precision_score(y_true[idx], y_proba[idx]) for idx in (np.random.randint(0, len(y_proba), len(y_proba)) for _ in range(n_bootstraps)) if len(np.unique(y_true[idx])) > 1]
        mean_ap, ci_ap = np.mean(boot_aps), np.percentile(boot_aps, [2.5, 97.5])
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=f'{name}\nAUPRC={mean_ap:.3f} (CI {ci_ap[0]:.3f}-{ci_ap[1]:.3f})', ax=ax_pr)
    ax_pr.set_title('PR Curve Comparison', fontsize=14, weight='bold')

    # Full Enrichment Plot
    ax_enr = axes[1, 0]
    for name, data in cv_prediction_data.items():
        df_enrich = pd.DataFrame({'y_true': data['y_true'], 'y_proba': data['y_pred_proba']}).sort_values('y_proba', ascending=False)
        y_enrich = df_enrich['y_true'].cumsum() / df_enrich['y_true'].sum() * 100
        x_enrich = np.arange(1, len(df_enrich) + 1) / len(df_enrich) * 100
        ax_enr.plot(x_enrich, y_enrich, label=name, lw=2)
    ax_enr.plot([0, 100], [0, 100], 'k--', label='Random'); ax_enr.set_title('Enrichment Plot (Full)', fontsize=14, weight='bold')
    ax_enr.set_xlabel('% Library Screened'); ax_enr.set_ylabel('% Actives Found'); ax_enr.legend()

    # Early Enrichment Plot
    ax_zoom = axes[1, 1]
    for name, data in cv_prediction_data.items():
        df_enrich = pd.DataFrame({'y_true': data['y_true'], 'y_proba': data['y_pred_proba']}).sort_values('y_proba', ascending=False)
        y_enrich = df_enrich['y_true'].cumsum() / df_enrich['y_true'].sum() * 100
        x_enrich = np.arange(1, len(df_enrich) + 1) / len(df_enrich) * 100
        ax_zoom.plot(x_enrich, y_enrich, label=name, lw=2)
    ax_zoom.plot([0, 20], [0, 20], 'k--', label='Random'); ax_zoom.set_title('Early Enrichment (Top 20%)', fontsize=14, weight='bold')
    ax_zoom.set_xlabel('% Library Screened'); ax_zoom.set_ylabel('% Actives Found'); ax_zoom.set_xlim(0, 20); ax_zoom.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]);
    pdf_path = os.path.join(output_dir, 'pipeline_cv_curves_comparison.pdf'); plt.savefig(pdf_path, dpi=300)
    plt.show()
    print(f"Combined CV plots saved to {pdf_path}")

    # ================================================================
    # --- NEW: Part 2: Confusion Matrix Plots (Dynamic Grid) ---
    # ================================================================
    print("\n--- Generating Combined Confusion Matrices (Threshold 0.5) ---")

    n_methods = len(cv_prediction_data)
    if n_methods == 0:
        print("No data to plot confusion matrices.")
        return

    # Calculate grid size (e.g., 3 methods -> 2x2, 4 methods -> 2x2, 5 methods -> 2x3)
    ncols = int(np.ceil(np.sqrt(n_methods)))
    nrows = int(np.ceil(n_methods / ncols))

    fig_cm, axes_cm = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5.5))
    fig_cm.suptitle('Cross-Validated Confusion Matrices (Threshold 0.5)', fontsize=18, weight='bold')

    # Flatten axes array for easy iteration
    axes_flat = axes_cm.flatten()

    for i, (name, data) in enumerate(cv_prediction_data.items()):
        ax = axes_flat[i]
        y_true = data['y_true']

        # --- IMPORTANT ---
        # Convert probabilities to binary predictions using a 0.5 threshold
        y_pred = (data['y_pred_proba'] > 0.5).astype(int)
        # ---

        # Plot confusion matrix
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            ax=ax,
            cmap=plt.cm.Blues,
            values_format='d'
        )
        ax.set_title(f'Confusion Matrix: {name}', fontsize=12, weight='bold')

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    cm_pdf_path = os.path.join(output_dir, 'pipeline_cv_confusion_matrices.pdf')
    plt.savefig(cm_pdf_path, dpi=300)
    plt.show()
    print(f"Combined confusion matrices saved to {cm_pdf_path}")

plot_combined_pipeline_curves(pipeline_cv_predictions, output_dir=output_path, n_bootstraps=N_BOOTSTRAPS)


#%% md
# ## 7. Final Evaluation on Holdout Test Set
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker  # Import the ticker module
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay

def generate_final_evaluation_plots(
    model,
    X_test,
    y_test,
    model_name="Final Model",
    output_dir="."
):
    """
    Generates a suite of final evaluation plots with a more detailed enrichment plot
    and saves them to a single multi-page PDF.
    """
    print("\n--- Generating Final Evaluation Plots ---")

    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pdf_path = os.path.join(output_dir, f'final_{model_name.replace(" ", "_")}_plots.pdf')

    with PdfPages(pdf_path) as pdf:
        # --- 1. ROC Curve ---
        fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=model_name, ax=ax_roc)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        ax_roc.set_title(f'Final Model ROC Curve ({model_name})', fontsize=14, weight='bold')
        pdf.savefig(fig_roc, bbox_inches='tight')
        plt.close(fig_roc)

        # --- 2. Precision-Recall Curve ---
        fig_pr, ax_pr = plt.subplots(figsize=(8, 8))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=model_name, ax=ax_pr)
        baseline = y_test.mean()
        ax_pr.axhline(baseline, ls='--', color='k', label=f'No Skill ({baseline:.2f})')
        ax_pr.set_title(f'Final Model PR Curve ({model_name})', fontsize=14, weight='bold')
        pdf.savefig(fig_pr, bbox_inches='tight')
        plt.close(fig_pr)

        # --- 3. Confusion Matrix ---
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
        ax_cm.set_title(f'Final Model Confusion Matrix ({model_name})', fontsize=14, weight='bold')
        ax_cm.set_xlabel('Predicted Label'); ax_cm.set_ylabel('True Label')
        pdf.savefig(fig_cm, bbox_inches='tight')
        plt.close(fig_cm)

        # --- 4. Enrichment Plot (UPDATED with more detail) ---
        fig_enr, ax_enr = plt.subplots(figsize=(8, 8))

        # Prepare data
        df_enrich = pd.DataFrame({'y_true': y_test, 'y_proba': y_pred_proba}).sort_values('y_proba', ascending=False)
        df_enrich['cumulative_actives'] = df_enrich['y_true'].cumsum()
        total_actives = df_enrich['y_true'].sum()
        x_enrich = np.arange(1, len(df_enrich) + 1) / len(df_enrich) * 100
        y_enrich = df_enrich['cumulative_actives'] / total_actives * 100

        # Plot curves
        ax_enr.plot(x_enrich, y_enrich, label=f'{model_name} Enrichment', lw=2.5)
        ax_enr.plot([0, 100], [0, 100], 'k--', label='Random Selection')

        # Find the y-value at x=10%
        idx_10_pct = np.searchsorted(x_enrich, 10)
        if idx_10_pct < len(y_enrich):
            y_10_pct = y_enrich.iloc[idx_10_pct]

            # Add dotted lines to axes
            ax_enr.plot([10, 10], [0, y_10_pct], 'r:', lw=1.5)
            ax_enr.plot([0, 10], [y_10_pct, y_10_pct], 'r:', lw=1.5)

            # Add annotation
            ax_enr.annotate(f'Top 10% screened finds\n{y_10_pct:.1f}% of total actives',
                            xy=(10, y_10_pct),
                            xytext=(15, y_10_pct - 15),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="black", lw=1, alpha=0.7))

        ax_enr.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_enr.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax_enr.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_enr.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax_enr.grid(which='both', linestyle=':')

        # Set titles and labels
        ax_enr.set_title('Final Model Enrichment Plot', fontsize=14, weight='bold')
        ax_enr.set_xlabel('% of Library Screened (Ranked by Score)', fontsize=12)
        ax_enr.set_ylabel('% of Total Actives Found', fontsize=12)
        ax_enr.set_xlim(0, 100); ax_enr.set_ylim(0, 100)
        ax_enr.legend()
        pdf.savefig(fig_enr, bbox_inches='tight')
        plt.close(fig_enr)

    print(f"\n✅ Final evaluation plots saved to: {pdf_path}")

#%%
# --- Get winning model info from the results dataframe ---
best_pipeline_name = results_df.iloc[0]['Pipeline']
best_feature_method_name = results_df.iloc[0]['Feature Method']
best_params = results_df.iloc[0]['Best Params']

print(f"--- Final evaluation of '{best_pipeline_name}' on the holdout test set ---")

# --- Get feature sets ---
best_ft_key = next(item['key'] for item in feature_configs if item["name"] == best_feature_method_name)
X_train_final = X_train_sets[best_ft_key]

# Generate features for the test set
X_test_final_cache = os.path.join(cache_path, f"X_test_final_{best_ft_key}.parquet")
if os.path.exists(X_test_final_cache):
    X_test_final = pd.read_parquet(X_test_final_cache)
else:
    best_ft_params = next(item.get('params', {}) for item in feature_configs if item["name"] == best_feature_method_name)
    X_test_final = get_features(df_test, method=best_ft_key, **best_ft_params)
    X_test_final.to_parquet(X_test_final_cache)

# --- Train the FINAL TUNED model (no calibration) ---
print("\nTraining final model with best parameters...")
final_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
final_model.fit(X_train_final, y_train)

# --- Generate and save all requested plots ---
generate_final_evaluation_plots(
    model=final_model,
    X_test=X_test_final,
    y_test=y_test,
    model_name=best_pipeline_name,
    output_dir=output_path
)

# --- Print final metrics to console ---
y_pred_final = final_model.predict(X_test_final)
y_proba_final = final_model.predict_proba(X_test_final)[:, 1]
final_auc = roc_auc_score(y_test, y_proba_final)

print(f"\nFinal Holdout ROC AUC: {final_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Inactive', 'Active']))
#%%
