# matrix_based_split.py

import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Levenshtein import ratio
from joblib import Parallel, delayed

class SimilarityMatrix:
    """
    Pre-computes and stores a pairwise similarity matrix for a set of sequences.
    This is highly optimized for analyses where the same dataset is split multiple
    times with different parameters, as the expensive similarity calculation
    is only performed once.
    """
    def __init__(self, sequences, n_jobs=-1, verbose=True):
        self.sequences = np.array(sequences)
        self.n_sequences = len(sequences)
        self.verbose = verbose

        if verbose:
            print(f"Computing similarity matrix for {self.n_sequences} sequences...")
        start_time = time.time()
        self.matrix = self._compute_similarity_matrix(n_jobs)
        if verbose:
            print(f"Similarity matrix computed in {time.time() - start_time:.2f}s")

    def _compute_similarity_matrix(self, n_jobs):
        """Computes the matrix in parallel."""
        indices = np.triu_indices(self.n_sequences, k=1)
        
        def compute_batch(i_vals, j_vals):
            return [ratio(self.sequences[i], self.sequences[j]) for i, j in zip(i_vals, j_vals)]

        # Batch processing for parallel jobs
        n_pairs = len(indices[0])
        batch_size = int(np.ceil(n_pairs / abs(n_jobs))) if n_jobs != 0 else n_pairs
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_batch)(indices[0][i:i+batch_size], indices[1][i:i+batch_size])
            for i in range(0, n_pairs, batch_size)
        )
        
        similarities = [item for sublist in results for item in sublist]
        
        matrix = np.zeros((self.n_sequences, self.n_sequences), dtype=np.float32)
        matrix[indices] = similarities
        matrix += matrix.T
        np.fill_diagonal(matrix, 1.0)
        return matrix

    def save(self, filepath):
        """Saves the matrix and sequences to a compressed .npz file."""
        np.savez_compressed(filepath, matrix=self.matrix, sequences=self.sequences)
        if self.verbose:
            print(f"Similarity matrix saved to {filepath}")

    @classmethod
    def load(cls, filepath, verbose=True):
        """Loads a SimilarityMatrix from a .npz file."""
        with np.load(filepath, allow_pickle=True) as data:
            instance = cls.__new__(cls)
            instance.matrix = data['matrix']
            instance.sequences = data['sequences']
            instance.n_sequences = len(instance.sequences)
            instance.verbose = verbose
            if verbose:
                print(f"Similarity matrix loaded from {filepath}")
            return instance

def homology_aware_split(similarity_matrix, y, test_size=0.2, similarity_threshold=0.8, random_state=42, verbose=True):
    """
    Creates a train-test split that prevents data leakage by keeping similar sequences
    (homologs) in the same set. This is the definitive method for this task.

    How it works:
    1.  A graph is built where sequences are nodes and an edge exists if their
        similarity is >= similarity_threshold.
    2.  It finds all connected components (unbreakable clusters) in this graph.
    3.  It intelligently assigns these entire clusters to the train or test set
        to best honor the requested `test_size`, using the `random_state` for
        reproducibility.
    """
    if verbose:
        print("\nPerforming homology-aware graph-based split...")
        print(f"Similarity threshold: {similarity_threshold * 100:.0f}%")

        # 1. Build graph and find clusters
    adj_matrix = similarity_matrix.matrix >= similarity_threshold
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    components = [np.where(labels == i)[0] for i in range(n_components)]

    # --- NEW: Get cluster metrics for analysis ---
    if not components:
        largest_component_size = 0
    else:
        largest_component_size = max(len(c) for c in components)

    if verbose:
        print(f"Found {n_components} clusters. Largest cluster has {largest_component_size} members.")

    # 2. Intelligent cluster assignment (remains the same)
    target_test_count = int(len(y) * test_size)
    train_indices, test_indices = [], []
    remaining_components = []
    for comp in components:
        if len(comp) > target_test_count:
            train_indices.extend(comp)
        else:
            remaining_components.append(comp)

    rng = np.random.default_rng(random_state)
    rng.shuffle(remaining_components)

    for comp in remaining_components:
        if len(test_indices) + len(comp) <= target_test_count:
            test_indices.extend(comp)
        else:
            train_indices.extend(comp)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # 3. Final report
    actual_test_size = len(test_indices) / len(y) if len(y) > 0 else 0
    if verbose:
        print("\nHomology-aware split completed:")
        print(f"  Train set: {len(train_indices)} ({100 - actual_test_size*100:.1f}%)")
        print(f"  Test set: {len(test_indices)} ({actual_test_size*100:.1f}%)")
        
        # Final check for violations (should be 0)
        test_train_similarities = similarity_matrix.matrix[np.ix_(test_indices, train_indices)]
        violations = np.sum(np.any(test_train_similarities >= similarity_threshold, axis=1))
        print(f"  Final violations at threshold: {violations}")
        
    return train_indices, test_indices, n_components, largest_component_size