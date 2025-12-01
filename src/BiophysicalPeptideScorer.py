import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from joblib import Parallel, delayed
import time


class BiophysicalPeptideScorer:
    """
    Advanced biophysical scoring function for membrane-active peptides that combines
    machine learning predictions with biophysical properties and their interactions.
    Assumes hydrophobic moment (amphipathicity) is provided as a column.
    """

    def __init__(self,
                 # Column names for properties in DataFrame
                 gravy_column='gravy',
                 helicity_column='helicity',
                 charge_column='charge',
                 hydrophobic_moment_column='hydrophobic_moment',

                 # Model parameters - these can be optimized
                 w_prob=0.5,  # Weight for ML probability
                 charge_opt_orig=1.5,  # Optimal charge
                 charge_width_orig=1.0,  # Charge width parameter

                 helicity_threshold_orig=0.38,  # Helicity threshold
                 helicity_steepness_orig=8.0,  # Helicity sigmoid steepness

                 amphipathicity_threshold_orig=0.52,  # Amphipathicity threshold
                 amphipathicity_steepness_orig=4.0,  # Amphipathicity sigmoid steepness

                 gravy_threshold_orig=0.8,  # Minimum effective GRAVY
                 gravy_steepness_orig = 8.0,  # How sharply effect increases around threshold

                 coop_threshold=0.4,  # Cooperative threshold
                 boost_weight=0.3,  # Cooperative boost weight

                 # Component weights
                 w_charge=0.1,  # Weight for charge score
                 w_helicity=0.1,  # Weight for helicity effect
                 w_gravy=0.1,  # Weight for GRAVY impact
                 w_amphipathicity=0.1,  # Weight for amphipathicity score
                 w_cooperative=0.1,  # Weight for cooperative effect

                 # Calculation options
                 normalize_properties=False  # Whether to normalize properties
                 ):

        # Store column names
        self.gravy_column = gravy_column
        self.helicity_column = helicity_column
        self.charge_column = charge_column
        self.hydrophobic_moment_column = hydrophobic_moment_column

        # Store model parameters
        self.w_prob = w_prob
        self.charge_opt_orig = charge_opt_orig
        self.charge_width_orig = charge_width_orig
        self.helicity_threshold_orig = helicity_threshold_orig
        self.helicity_steepness_orig = helicity_steepness_orig
        self.amphipathicity_threshold_orig = amphipathicity_threshold_orig
        self.amphipathicity_steepness_orig = amphipathicity_steepness_orig
        self.gravy_threshold_orig = gravy_threshold_orig
        self.gravy_steepness_orig = gravy_steepness_orig
        #self.base_impact = base_impact
        self.coop_threshold = coop_threshold
        self.boost_weight = boost_weight

        # Store component weights
        self.w_charge = w_charge
        self.w_helicity = w_helicity
        self.w_gravy = w_gravy
        self.w_amphipathicity = w_amphipathicity
        self.w_cooperative = w_cooperative

        # Store calculation options
        self.normalize_properties = normalize_properties

        # Initialize scalers
        if normalize_properties:
            self.gravy_scaler = MinMaxScaler()
            self.helicity_scaler = MinMaxScaler()
            self.charge_scaler = MinMaxScaler()
            self.amphipathicity_scaler = MinMaxScaler()

            # Transformed parameters will be set after fitting
            self.charge_opt = None
            self.charge_width = None
            self.helicity_threshold = None
            self.helicity_steepness = None
            self.gravy_threshold = None
            self.gravy_steepness = None
            self.amphipathicity_threshold = None
            self.amphipathicity_steepness = None
            self.coop_threshold = self.coop_threshold  # This stays the same
        else:
            # If not normalizing, use original parameters directly
            self.charge_opt = self.charge_opt_orig
            self.charge_width = self.charge_width_orig
            self.helicity_threshold = self.helicity_threshold_orig
            self.helicity_steepness = self.helicity_steepness_orig
            self.gravy_threshold = self.gravy_threshold_orig
            self.gravy_steepness = self.gravy_steepness_orig
            self.amphipathicity_threshold = self.amphipathicity_threshold_orig
            self.amphipathicity_steepness = self.amphipathicity_steepness_orig
            self.coop_threshold = self.coop_threshold

    def fit(self, X, y=None):
        """
        Fit the scalers on training data

        Parameters:
        -----------
        X : DataFrame
            DataFrame containing peptide properties
        y : array-like, optional
            Target values (not used, included for compatibility)

        Returns:
        --------
        self
        """
        # Create normalized copies of the properties
        if self.normalize_properties:

            self.gravy_scaler.fit(X[self.gravy_column].values.reshape(-1, 1))
            self.helicity_scaler.fit(X[self.helicity_column].values.reshape(-1, 1))
            self.charge_scaler.fit(X[self.charge_column].values.reshape(-1, 1))
            self.amphipathicity_scaler.fit(X[self.hydrophobic_moment_column].values.reshape(-1, 1))

            # Also normalize the thresholds using the same scalers
            gravy_threshold = self.gravy_scaler.transform([[self.gravy_threshold_orig]])[0][0]
            helicity_threshold = self.helicity_scaler.transform([[self.helicity_threshold_orig]])[0][0]
            amphipathicity_threshold = self.amphipathicity_scaler.transform([[self.amphipathicity_threshold_orig]])[0][0]

            # For charge, we need to transform both the optimal value and width
            charge_opt = self.charge_scaler.transform([[self.charge_opt_orig]])[0][0]

            # Width needs to be scaled by the range of the data
            charge_min = self.charge_scaler.data_min_[0]
            charge_max = self.charge_scaler.data_max_[0]
            charge_width = self.charge_width_orig / (charge_max - charge_min)

            # Steepness parameters need inverse scaling (multiply by range)
            helicity_min = self.helicity_scaler.data_min_[0]
            helicity_max = self.helicity_scaler.data_max_[0]
            helicity_steepness = self.helicity_steepness_orig * (helicity_max - helicity_min)

            gravy_min = self.gravy_scaler.data_min_[0]
            gravy_max = self.gravy_scaler.data_max_[0]
            gravy_steepness = self.gravy_steepness_orig * (gravy_max - gravy_min)

            amphipathicity_min = self.amphipathicity_scaler.data_min_[0]
            amphipathicity_max = self.amphipathicity_scaler.data_max_[0]
            amphipathicity_steepness = self.amphipathicity_steepness_orig * (amphipathicity_max - amphipathicity_min)

        return self

    def calculate_scores(self, X, probabilities):
        """
        Calculate the biophysical scores for peptides

        Parameters:
        -----------
        X : DataFrame
            DataFrame containing peptide properties
        probabilities : array-like
            Probabilities from machine learning model

        Returns:
        --------
        dict
            Dictionary with all component scores and final score
        """
        # Extract properties
        gravy = X[self.gravy_column].values
        helicity = X[self.helicity_column].values
        charge = X[self.charge_column].values
        amphipathicity = X[self.hydrophobic_moment_column].values

        # Normalize values if needed
        if self.normalize_properties:
            gravy = self.gravy_scaler.transform(gravy.reshape(-1, 1)).flatten()
            helicity = self.helicity_scaler.transform(helicity.reshape(-1, 1)).flatten()
            amphipathicity = self.amphipathicity_scaler.transform(amphipathicity.reshape(-1, 1)).flatten()
            charge = self.charge_scaler.transform(charge.reshape(-1, 1)).flatten()

        # 1. Calculate Charge Score (Gaussian function)
        charge_score = np.exp(-((charge - self.charge_opt_orig) ** 2) / (2 * self.charge_width_orig ** 2))

        # 2. Calculate Helicity Effect (Sigmoid function)
        helicity_effect = 1 / (1 + np.exp(-self.helicity_steepness_orig * (helicity - self.helicity_threshold_orig)))

        # 3. NEW - Calculate Direct GRAVY Effect (Sigmoid function)
        # Parameters based on biophysical knowledge

        gravy_effect = 1 / (1 + np.exp(-self.gravy_steepness_orig * (gravy - self.gravy_threshold_orig)))


        # 5. Calculate Amphipathicity Score
        amphipathicity_score = 1 / (1 + np.exp(-self.amphipathicity_steepness_orig *
                                               (amphipathicity - self.amphipathicity_threshold_orig)))

        # 6. Calculate Cooperative Effect
        cooperative_effect = np.minimum.reduce([charge_score, helicity_effect,
                                                gravy_effect, amphipathicity_score])

        # 7. Calculate Cooperative Boost
        cooperative_boost = self.boost_weight * (
                (charge_score > self.coop_threshold) &
                (helicity_effect > self.coop_threshold) &
                (gravy_effect > self.coop_threshold) &
                (amphipathicity_score > self.coop_threshold)
        ).astype(float)

        # 8. Calculate Biophysical Score
        raw_biophysical_score = (
                self.w_charge * charge_score +
                self.w_helicity * helicity_effect +
                self.w_gravy * gravy_effect +
                self.w_amphipathicity * amphipathicity_score +
                self.w_cooperative * cooperative_effect +
                cooperative_boost
        )

        # Calculate maximum possible score
        max_possible_score = (
                self.w_charge + self.w_helicity + self.w_gravy +
                self.w_amphipathicity + self.w_cooperative + self.boost_weight
        )

        # Normalize if the maximum possible exceeds 1.0
        biophysical_score = raw_biophysical_score / max_possible_score


        # 9. Calculate Final Score
        final_score = self.w_prob * probabilities + (1 - self.w_prob) * biophysical_score

        # Create results dictionary
        results = {
            'charge_score': charge_score,
            'helicity_effect': helicity_effect,
            'gravy_effect': gravy_effect,
            'amphipathicity_score': amphipathicity_score,
            'cooperative_effect': cooperative_effect,
            'cooperative_boost': cooperative_boost,
            'biophysical_score': biophysical_score,
            'original_probability': probabilities,
            'final_score': final_score
        }

        return results

    def predict(self, X, model):
        """
        Generate predictions using both the RF model and biophysical scoring

        Parameters:
        -----------
        X : DataFrame
            DataFrame containing peptide properties
        model : trained model
            Trained machine learning model with predict_proba method

        Returns:
        --------
        tuple
            (predictions, scores_df) - binary predictions and detailed scores
        """
        # Get probabilities from the model
        probabilities = model.predict_proba(X)[:, 1]

        # Calculate scores
        scores = self.calculate_scores(X, probabilities)

        # Convert to DataFrame
        scores_df = pd.DataFrame(scores)

        # Generate binary predictions using threshold of 0.5
        predictions = (scores_df['final_score'] >= 0.5).astype(int)

        return predictions, scores_df

    def rank_peptides(self, X, model):
        """
        Rank peptides based on final score

        Parameters:
        -----------
        X : DataFrame
            DataFrame containing peptide properties
        model : trained model
            Trained machine learning model with predict_proba method

        Returns:
        --------
        DataFrame
            Original DataFrame with scores and rankings
        """
        # Get probabilities from the model
        probabilities = model.predict_proba(X)[:, 1]

        # Calculate scores
        scores = self.calculate_scores(X, probabilities)

        # Convert to DataFrame
        scores_df = pd.DataFrame(scores)

        # Combine with original data
        result_df = X.copy()
        for col in scores_df.columns:
            result_df[col] = scores_df[col]

        # Add rank based on final score
        result_df['rank'] = result_df['final_score'].rank(ascending=False, method='min')

        # Sort by rank
        result_df = result_df.sort_values('rank')

        return result_df

    @classmethod
    def grid_search_optimizer(cls, X, y, model, param_grid=None, cv=5, scoring='top10_precision', n_jobs=-1,
                              verbose=1):
        """
        Perform grid search to find optimal parameters for the biophysical scoring function.

        Parameters:
        -----------
        X : DataFrame
            Features DataFrame containing peptide properties
        y : Series or array
            Target values (active/inactive)
        model : estimator
            Trained machine learning model with predict_proba method
        param_grid : dict, default=None
            Dictionary with parameters names as keys and lists of parameter values to try
            If None, a default parameter grid will be used
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='top10_precision'
            Metric to optimize. Options:
            - 'auc': Area under ROC curve
            - 'avg_precision': Average precision score
            - 'top1_precision': Precision in top 1%
            - 'top5_precision': Precision in top 5%
            - 'top10_precision': Precision in top 10%
            - 'top20_precision': Precision in top 20%
            - 'f1': F1 score
        n_jobs : int, default=-1
            Number of jobs to run in parallel (-1 means using all processors)
        verbose : int, default=1
            Controls the verbosity: higher values mean more messages

        Returns:
        --------
        dict
            Dictionary with best parameters and results
        """

        # Define default parameter grid if not provided
        if param_grid is None:
            param_grid = {
                'w_prob': [0.3, 0.4, 0.5, 0.6, 0.7],
                'charge_width_orig': [1.0, 2.0, 3.0],
                'helicity_steepness_orig': [8.0, 12.00, 16.0],
                'amphipathicity_steepness_orig': [4.0, 8.0],
                'gravy_steepness_orig': [8.0, 12.00, 16.0],
                'coop_threshold': [0.4, 0.5],
                'boost_weight': [0.3, 0.4],
                'w_charge': [0.1, 0.2, 0.3],  # Weight for charge score
                'w_helicity':[0.1, 0.2, 0.3],  # Weight for helicity effect
                'w_gravy': [0.1, 0.2, 0.3],  # Weight for GRAVY impact
                'w_amphipathicity': [0.1, 0.2, 0.3],  # Weight for amphipathicity score
                'w_cooperative': [0.1, 0.2, 0.3],  # Weight for cooperative effect
            }

        # Create all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))

        if verbose >= 1:
            print(f"Grid search with {len(param_combinations)} parameter combinations")
            print(f"Cross-validation with {cv} folds")
            print(f"Optimizing for metric: {scoring}")

        # Function to evaluate a parameter combination
        def evaluate_params(params, X, y, model, cv_splits):
            # Create a scorer with these parameters
            scorer = BiophysicalPeptideScorer(**params)

            # Store scores for each fold
            fold_scores = []

            # Cross-validation
            for train_idx, val_idx in cv_splits:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Fit scorer on training data
                scorer.fit(X_train)

                # Get probabilities from model
                val_probs = model.predict_proba(X_val)[:, 1]

                # Calculate scores
                scores = scorer.calculate_scores(X_val, val_probs)
                final_scores = scores['final_score']
                predictions = (final_scores >= 0.5).astype(int)

                # Calculate requested metric
                if scoring == 'auc':
                    score = roc_auc_score(y_val, final_scores)
                elif scoring == 'avg_precision':
                    score = average_precision_score(y_val, final_scores)
                elif scoring == 'f1':
                    score = f1_score(y_val, predictions)
                elif scoring.startswith('top'):
                    # Extract percentage from scoring name (e.g., 'top10_precision' -> 10)
                    pct = int(scoring.split('top')[1].split('_')[0]) / 100
                    n_top = max(1, int(len(y_val) * pct))

                    # Get top indices by score
                    top_indices = np.argsort(-final_scores)[:n_top]

                    # Calculate precision in top N%
                    score = np.mean(y_val.iloc[top_indices])
                else:
                    raise ValueError(f"Unknown scoring metric: {scoring}")

                fold_scores.append(score)

            # Average score across folds
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            return {
                'params': params,
                'score_mean': avg_score,
                'score_std': std_score
            }

        # Create cross-validation splits
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_splits = list(cv_splitter.split(X))

        # Start timer
        start_time = time.time()

        # Run evaluations in parallel
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_params)(params, X, y, model, cv_splits)
            for params in param_combinations
        )

        # Find best parameters
        best_idx = np.argmax([r['score_mean'] for r in results])
        best_result = results[best_idx]

        # Stop timer
        elapsed_time = time.time() - start_time

        if verbose >= 1:
            print("\nGrid search completed in {:.2f} seconds".format(elapsed_time))
            print("\nBest parameters:")
            for param, value in best_result['params'].items():
                print(f"  {param}: {value}")
            print(f"\nBest {scoring} score: {best_result['score_mean']:.4f} (±{best_result['score_std']:.4f})")

        # Create a DataFrame with all results for analysis
        results_df = pd.DataFrame([{**r['params'], 'score': r['score_mean'], 'std': r['score_std']} for r in results])

        # Sort by score
        results_df = results_df.sort_values('score', ascending=False)

        return {
            'best_params': best_result['params'],
            'best_score': best_result['score_mean'],
            'best_score_std': best_result['score_std'],
            'all_results': results_df,
            'elapsed_time': elapsed_time
        }


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Example data (replace with your actual data)
    data = {
        'gravy': [0.2, -0.1, 0.7],
        'helicity': [0.7, 0.5, 0.3],
        'charge': [5, 4, 2],
        'hydrophobic_moment': [0.5, 0.4, 0.2],
        'active': [1, 1, 0]
    }

    df = pd.DataFrame(data)

    # Split data
    X = df.drop('active', axis=1)
    y = df['active']

    # Train a simple RF model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Create and fit the scorer
    scorer = BiophysicalPeptideScorer(
        gravy_column='gravy',
        helicity_column='helicity',
        charge_column='charge',
        hydrophobic_moment_column='hydrophobic_moment'
    )
    scorer.fit(X)

    # Rank peptides
    ranked_peptides = scorer.rank_peptides(X, model)

    # Print results
    print("Ranked peptides with scores:")
    print(ranked_peptides[['gravy', 'helicity', 'charge', 'hydrophobic_moment',
                           'original_probability', 'final_score', 'rank']])


    # Function to demonstrate parameter optimization
    def optimize_parameters(X_train, y_train, X_val, y_val):
        """Simple grid search for parameter optimization"""
        param_grid = {
            'charge_opt': [4, 5, 6],
            'helicity_threshold': [0.3, 0.4, 0.5],
            'w_prob': [0.4, 0.5, 0.6]
        }

        # Train a model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        best_score = 0
        best_params = {}

        # Simple grid search
        for charge_opt in param_grid['charge_opt']:
            for helicity_threshold in param_grid['helicity_threshold']:
                for w_prob in param_grid['w_prob']:
                    # Create scorer with these parameters
                    scorer = BiophysicalPeptideScorer(
                        charge_opt_orig=charge_opt,
                        helicity_threshold_orig=helicity_threshold,
                        w_prob=w_prob
                    )
                    scorer.fit(X_train)

                    # Rank validation peptides
                    ranked = scorer.rank_peptides(X_val, model)

                    # Evaluate using precision in top 30%
                    top_n = max(1, int(len(y_val) * 0.3))
                    top_indices = ranked.index[:top_n]
                    precision = y_val.loc[top_indices].mean()

                    if precision > best_score:
                        best_score = precision
                        best_params = {
                            'charge_opt': charge_opt,
                            'helicity_threshold': helicity_threshold,
                            'w_prob': w_prob
                        }

        print(f"Best parameters: {best_params}")
        print(f"Best precision at top 30%: {best_score:.4f}")

        return best_params