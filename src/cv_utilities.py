"""
Cross-validation utilities including custom splitters and experiment runners.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.sparse.csgraph import connected_components
from BiophysicalPeptideScorer import BiophysicalPeptideScorer


class HomologyAwareCV:
    """
    Custom cross-validation splitter that respects sequence homology.
    Ensures homologous sequences stay together in train/val splits.
    """
    
    def __init__(self, similarity_matrix, original_indices, threshold, n_splits=5, random_state=42):
        """
        Parameters:
        -----------
        similarity_matrix : SimilarityMatrix
            Similarity matrix object with .matrix attribute
        original_indices : np.ndarray
            Indices of training data in the full dataset
        threshold : float
            Similarity threshold for grouping homologous sequences
        n_splits : int
            Number of CV folds
        random_state : int
            Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.similarity_matrix = similarity_matrix
        self.original_indices = original_indices
        self.threshold = threshold

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate train/val fold indices respecting homology clusters.
        
        Yields:
        -------
        train_idx : np.ndarray
            Training indices for this fold
        val_idx : np.ndarray
            Validation indices for this fold
        """
        # Get submatrix for training data
        adj_matrix = self.similarity_matrix.matrix[
            np.ix_(self.original_indices, self.original_indices)
        ] >= self.threshold
        
        # Find connected components (homology clusters)
        n_components, labels = connected_components(
            csgraph=adj_matrix, directed=False, return_labels=True
        )
        
        # Group indices by component
        components = [
            np.where(labels == i)[0] for i in range(n_components)
        ]
        
        # Assign components to folds
        fold_assignments = np.arange(len(components)) % self.n_splits
        np.random.RandomState(self.random_state).shuffle(fold_assignments)
        
        # Yield train/val splits
        for i in range(self.n_splits):
            val_idx = np.concatenate([
                components[j] for j in np.where(fold_assignments == i)[0]
            ])
            train_idx = np.concatenate([
                components[j] for j in np.where(fold_assignments != i)[0]
            ])
            if len(val_idx) > 0 and len(train_idx) > 0:
                yield train_idx, val_idx


class RandomSplitCV:
    """Standard random K-fold cross-validation splitter."""
    
    def __init__(self, n_splits=5, random_state=42):
        """
        Parameters:
        -----------
        n_splits : int
            Number of CV folds
        random_state : int
            Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate random train/val fold indices.
        
        Yields:
        -------
        train_idx : np.ndarray
            Training indices for this fold
        val_idx : np.ndarray
            Validation indices for this fold
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(X):
            yield train_idx, val_idx


def run_cv_experiment(features, y_train, cv_splitter, model_type, random_state=42):
    """
    Run a complete cross-validation experiment and return aggregated predictions.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix
    y_train : pd.Series
        Label series
    cv_splitter : HomologyAwareCV or RandomSplitCV
        CV splitter object
    model_type : str
        Either 'rf' (Random Forest) or 'camper' (RF + BiophysicalPeptideScorer)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict with keys:
        - 'y_true': aggregated true labels (np.ndarray)
        - 'y_pred': aggregated predictions (np.ndarray)
        - 'fold_scores': AUC scores per fold (list)
        - 'mean_auc': mean CV AUC (float)
        - 'std_auc': standard deviation of CV AUC (float)
    """
    if model_type not in ['rf', 'camper']:
        raise ValueError(f"model_type must be 'rf' or 'camper', got {model_type}")
    
    all_y_true = []
    all_y_pred = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(features, y_train)):
        print(f"  Fold {fold + 1}/{cv_splitter.get_n_splits()}", end=" - ")
        
        # Split data
        X_train_fold = features.iloc[train_idx]
        X_val_fold = features.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        if model_type == 'rf':
            # Train Random Forest
            model_fold = RandomForestClassifier(
                random_state=random_state,
                n_estimators=1200,
                max_depth=None,
                max_features='sqrt',
                n_jobs=-1
            )
            model_fold.fit(X_train_fold, y_train_fold)
            y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]
            
        else:  # model_type == 'camper'
            # Train RF + BiophysicalPeptideScorer
            model_fold = RandomForestClassifier(
                random_state=random_state,
                n_estimators=1200,
                max_depth=None,
                max_features='sqrt',
                n_jobs=-1
            )
            model_fold.fit(X_train_fold, y_train_fold)
            
            scorer_fold = BiophysicalPeptideScorer(
                gravy_column='gravy',
                helicity_column='helicity',
                charge_column='charge',
                hydrophobic_moment_column='hydrophobic_moment',
                normalize_properties=False
            )
            scorer_fold.fit(X_train_fold)
            
            _, scores_df = scorer_fold.predict(X_val_fold, model_fold)
            y_pred_proba = scores_df['final_score'].values
        
        # Calculate AUC
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        fold_scores.append(auc)
        print(f"AUC: {auc:.4f}")
        
        # Aggregate predictions
        all_y_true.extend(y_val_fold.values)
        all_y_pred.extend(y_pred_proba)
    
    return {
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'fold_scores': fold_scores,
        'mean_auc': np.mean(fold_scores),
        'std_auc': np.std(fold_scores)
    }
