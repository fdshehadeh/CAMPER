"""
Enhanced feature extraction utilities for peptide analysis.
Supports three different feature extraction methods: RDKit, ProtBERT, and N-grams.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import gc
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# Enable progress bars for pandas operations
tqdm.pandas()

# RDKit imports (only loaded when needed)
def _load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem
        return Chem, Descriptors, AllChem
    except ImportError:
        raise ImportError("RDKit is required for 'rdkit' method. Install with: conda install -c rdkit rdkit")

# Bio imports (only loaded when needed)
def _load_bio():
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        import peptides
        return ProteinAnalysis, peptides
    except ImportError:
        raise ImportError("Biopython and peptides are required for 'rdkit' method. Install with: pip install biopython peptides")

# Transformers imports (only loaded when needed)
def _load_transformers():
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        return AutoTokenizer, AutoModel, torch
    except ImportError:
        raise ImportError("Transformers and PyTorch are required for 'protbert' method. Install with: pip install transformers torch")


# RDKit-based functions (your original approach)
def get_mol(sequence):
    """Create an RDKit molecule from a peptide sequence"""
    Chem, _, _ = _load_rdkit()
    try:
        mol = Chem.MolFromSequence(sequence)
        return mol
    except Exception as e:
        print(f"Error creating molecule for sequence {sequence}: {e}")
        return None


def safe_calculate_descriptor(mol, func, name):
    """Safely calculate a descriptor for a molecule with error handling"""
    if mol is None:
        return None
    try:
        return func(mol)
    except Exception as e:
        print(f"Error calculating {name}: {e}")
        return None


def safe_calculate_fingerprint(mol):
    """Safely calculate Morgan fingerprint with error handling"""
    _, _, AllChem = _load_rdkit()
    if mol is None:
        return [0] * 2048  # Empty fingerprint
    try:
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=2048))
    except Exception as e:
        print(f"Error generating Morgan fingerprint: {e}")
        return [0] * 2048


def safe_calculate_helicity(seq):
    if seq is None:
        return None
    try:
        ProteinAnalysis, _ = _load_bio()
        return ProteinAnalysis(seq).secondary_structure_fraction()[0]
    except Exception as e:
        print(f"Error calculating Helicity: {e}")
        return None


def safe_calculate_gravy(seq):
    if seq is None:
        return None
    try:
        ProteinAnalysis, _ = _load_bio()
        return ProteinAnalysis(seq).gravy()
    except Exception as e:
        print(f"Error calculating Gravy: {e}")
        return None


def safe_calculate_charge(seq):
    if seq is None:
        return None
    try:
        ProteinAnalysis, _ = _load_bio()
        return ProteinAnalysis(seq).charge_at_pH(7.4)
    except Exception as e:
        print(f"Error calculating charge: {e}")
        return None


def safe_calculate_hydrophobic_moment(seq):
    if seq is None:
        return None
    try:
        _, peptides = _load_bio()
        return peptides.Peptide(seq).hydrophobic_moment(angle=100)
    except Exception as e:
        print(f"Error calculating hydrophobic_moment: {e}")
        return None


def get_rdkit_descriptors():
    """Get the RDKit descriptor dictionary"""
    _, Descriptors, _ = _load_rdkit()
    Chem, _, _ = _load_rdkit()
    
    descriptors = {
        'tpsa': Descriptors.TPSA,
        'MolLogP': Descriptors.MolLogP,
        'FractionCSP3': Descriptors.FractionCSP3,
        'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
        'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
        'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'NumAliphaticRings': Descriptors.NumAliphaticRings,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'RingCount': Descriptors.RingCount,
        'molar_refractivity': Chem.Crippen.MolMR,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'ExactMolWt': Descriptors.ExactMolWt,
        'MinPartialCharge': Descriptors.MinPartialCharge,
        'MaxPartialCharge': Descriptors.MaxPartialCharge,
        'BCUT2D_CHGHI': Descriptors.BCUT2D_CHGHI,
        'BCUT2D_CHGLO': Descriptors.BCUT2D_CHGLO,
        'LabuteASA': Descriptors.LabuteASA,
        'Kappa1': Descriptors.Kappa1,
        'Kappa2': Descriptors.Kappa2,
        'Kappa3': Descriptors.Kappa3,
        'BalabanJ': Descriptors.BalabanJ,
        'BertzCT': Descriptors.BertzCT,
        'SlogP_VSA6': Descriptors.SlogP_VSA6,
        'SlogP_VSA7': Descriptors.SlogP_VSA7,
        'SlogP_VSA8': Descriptors.SlogP_VSA8,
        'SlogP_VSA9': Descriptors.SlogP_VSA9,
        'SlogP_VSA10': Descriptors.SlogP_VSA10,
        'SlogP_VSA11': Descriptors.SlogP_VSA11,
        'SlogP_VSA12': Descriptors.SlogP_VSA12,
        "fr_quatN": Descriptors.fr_quatN,
        "fr_NH0": Descriptors.fr_NH0,
        "fr_NH1": Descriptors.fr_NH1,
        "fr_NH2": Descriptors.fr_NH2,
        "fr_guanido": Descriptors.fr_guanido,
        "fr_amidine": Descriptors.fr_amidine,
        "fr_N_O": Descriptors.fr_N_O,
        "fr_piperdine": Descriptors.fr_piperdine,
        "fr_piperzine": Descriptors.fr_piperzine,
        "fr_Ndealkylation1": Descriptors.fr_Ndealkylation1,
        "fr_Ndealkylation2": Descriptors.fr_Ndealkylation2,
        "fr_Nhpyrrole": Descriptors.fr_Nhpyrrole,
        "fr_unbrch_alkane": Descriptors.fr_unbrch_alkane,
        "fr_aryl_methyl": Descriptors.fr_aryl_methyl,
        "fr_alkyl_halide": Descriptors.fr_alkyl_halide,
        "fr_methoxy": Descriptors.fr_methoxy,
        "fr_benzene": Descriptors.fr_benzene,
        "fr_thiophene": Descriptors.fr_thiophene,
        "fr_furan": Descriptors.fr_furan,
        "fr_pyridine": Descriptors.fr_pyridine,
        "fr_dihydropyridine": Descriptors.fr_dihydropyridine,
        "fr_bicyclic": Descriptors.fr_bicyclic,
        "fr_thiazole": Descriptors.fr_thiazole,
        "fr_oxazole": Descriptors.fr_oxazole,
        "fr_imidazole": Descriptors.fr_imidazole,
        "fr_sulfide": Descriptors.fr_sulfide,
        "fr_thiocyan": Descriptors.fr_thiocyan,
        "fr_epoxide": Descriptors.fr_epoxide,
        "fr_term_acetylene": Descriptors.fr_term_acetylene,
        "fr_ketone_Topliss": Descriptors.fr_ketone_Topliss
    }
    return descriptors


def extract_rdkit_features(sequences, batch_size=1000):
    """Extract RDKit-based features (your original method)"""
    print("Extracting RDKit + Biophysical features...")
    
    descriptors = get_rdkit_descriptors()
    all_results = []
    n_samples = len(sequences)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Processing RDKit batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_sequences = sequences.iloc[start_idx:end_idx]

        # Create molecules
        batch_mols = batch_sequences.map(get_mol)

        # Initialize results for this batch
        batch_results = {}

        # Calculate descriptors
        for name, func in descriptors.items():
            batch_results[name] = batch_mols.map(
                lambda mol: safe_calculate_descriptor(mol, func, name)
            )

        # Calculate biophysical properties
        batch_results['gravy'] = batch_sequences.map(safe_calculate_gravy)
        batch_results['helicity'] = batch_sequences.map(safe_calculate_helicity)
        batch_results['charge'] = batch_sequences.map(safe_calculate_charge)
        batch_results['hydrophobic_moment'] = batch_sequences.map(safe_calculate_hydrophobic_moment)

        # Calculate Morgan fingerprints
        fingerprints = batch_mols.map(safe_calculate_fingerprint)

        # Create fingerprint DataFrame
        fp_df = pd.DataFrame(
            fingerprints.tolist(),
            index=batch_sequences.index,
            columns=[f'Morgan_fp_{i}' for i in range(2048)]
        )

        # Combine descriptor results
        desc_df = pd.DataFrame(batch_results, index=batch_sequences.index)

        # Combine all features
        batch_df = pd.concat([desc_df, fp_df], axis=1)
        all_results.append(batch_df)

        # Clear memory
        del batch_mols, batch_results, fp_df, desc_df, batch_df, fingerprints
        gc.collect()

    return pd.concat(all_results)


def extract_protbert_features(sequences, batch_size=32, max_length=512):
    """Extract ProtBERT embeddings"""
    print("Extracting ProtBERT features...")
    print("Loading ProtBERT model (this may take a while for first use)...")

    import requests
    from huggingface_hub import configure_http_backend

    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)
    AutoTokenizer, AutoModel, torch = _load_transformers()
    
    # Load ProtBERT model and tokenizer
    model_name = "Rostlab/prot_bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Set model to evaluation mode
    model.eval()

    # Check for available acceleration (MPS for Mac M1/M2, CUDA for NVIDIA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Metal Performance Shaders (MPS) acceleration")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA acceleration")
    else:
        device = torch.device('cpu')
        print("Using CPU (no acceleration available)")

    model.to(device)
    print(f"Using device: {device}")

    all_embeddings = []
    n_samples = len(sequences)
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Processing ProtBERT batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_sequences = sequences.iloc[start_idx:end_idx].tolist()

            # Add spaces between amino acids (ProtBERT format)
            formatted_sequences = [' '.join(seq) for seq in batch_sequences]

            # Tokenize sequences
            inputs = tokenizer(
                formatted_sequences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            # Get embeddings
            outputs = model(**inputs)

            # Use mean pooling over sequence length (excluding special tokens)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            # Calculate mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            # Move to CPU and convert to numpy
            batch_embeddings = mean_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings)

            # Clear GPU/MPS memory
            del inputs, outputs, token_embeddings, mean_embeddings
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Combine all embeddings
    embeddings_array = np.vstack(all_embeddings)

    # Create DataFrame with embedding columns
    embedding_columns = [f'protbert_dim_{i}' for i in range(embeddings_array.shape[1])]
    embeddings_df = pd.DataFrame(
        embeddings_array,
        index=sequences.index,
        columns=embedding_columns
    )

    print(f"Generated {embeddings_array.shape[1]} ProtBERT features")
    return embeddings_df


def generate_ngrams(sequence, n):
    """Generate n-grams from a sequence"""
    if len(sequence) < n:
        return []
    return [sequence[i:i+n] for i in range(len(sequence) - n + 1)]


def extract_ngram_features(sequences, min_freq=2):
    """Extract n-gram features (1-gram, 2-gram, 3-gram)"""
    print("Extracting N-gram features...")
    
    # Generate all n-grams for all sequences
    all_ngrams = []
    
    print("Generating n-grams...")
    for sequence in tqdm(sequences, desc="Processing sequences"):
        sequence_ngrams = []
        # Generate 1-grams, 2-grams, and 3-grams
        for n in range(1, 4):
            ngrams = generate_ngrams(sequence, n)
            sequence_ngrams.extend(ngrams)
        all_ngrams.append(' '.join(sequence_ngrams))
    
    # Use CountVectorizer to create binary features
    print("Creating n-gram feature matrix...")
    vectorizer = CountVectorizer(
        binary=True,  # Binary presence/absence
        min_df=min_freq,  # Minimum frequency
        token_pattern=r'\S+',  # Split on whitespace
        lowercase=False  # Keep original case
    )
    
    # Fit and transform
    ngram_matrix = vectorizer.fit_transform(all_ngrams)
    
    # Get feature names
    feature_names = [f'ngram_{name}' for name in vectorizer.get_feature_names_out()]
    
    # Create DataFrame
    ngram_df = pd.DataFrame(
        ngram_matrix.toarray(),
        index=sequences.index,
        columns=feature_names
    )
    
    print(f"Generated {len(feature_names)} n-gram features")
    return ngram_df


def get_features(df, method='rdkit', batch_size=1000, **kwargs):
    """
    Extract features from peptide sequences using different methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing a 'sequence' column with peptide sequences
    method : str, default='rdkit'
        Feature extraction method: 'rdkit', 'protbert', or 'ngrams'
    batch_size : int, default=1000
        Number of sequences to process in each batch (adjusted per method)
    **kwargs : dict
        Method-specific parameters:
        - For 'protbert': max_length (default=512)
        - For 'ngrams': min_freq (default=2)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with extracted features
    """
    if 'sequence' not in df.columns:
        raise ValueError("DataFrame must contain a 'sequence' column")
    
    sequences = df['sequence'].copy()
    
    if method == 'rdkit':
        return extract_rdkit_features(sequences, batch_size)
    
    elif method == 'protbert':
        # Use smaller batch size for ProtBERT due to memory requirements
        protbert_batch_size = min(batch_size, 32)
        max_length = kwargs.get('max_length', 512)
        return extract_protbert_features(sequences, protbert_batch_size, max_length)
    
    elif method == 'ngrams':
        min_freq = kwargs.get('min_freq', 2)
        return extract_ngram_features(sequences, min_freq)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'rdkit', 'protbert', 'ngrams'")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_sequences = [
        "WKLIAKLAKKLL",
        "ARLKWALKRLL",
        "GLLKRIKTLL",
        "KWKLFKKIPKFLHLAKKF"
    ]
    
    df_sample = pd.DataFrame({'sequence': sample_sequences})
    
    print("Testing different feature extraction methods...")
    
    # Test RDKit method
    try:
        print("\n=== Testing RDKit Method ===")
        features_rdkit = get_features(df_sample, method='rdkit', batch_size=2)
        print(f"RDKit features shape: {features_rdkit.shape}")
        print("First few columns:", list(features_rdkit.columns[:5]))
    except Exception as e:
        print(f"RDKit method failed: {e}")
    
    # Test N-grams method
    try:
        print("\n=== Testing N-grams Method ===")
        features_ngrams = get_features(df_sample, method='ngrams', min_freq=1)
        print(f"N-grams features shape: {features_ngrams.shape}")
        print("First few columns:", list(features_ngrams.columns[:5]))
    except Exception as e:
        print(f"N-grams method failed: {e}")
    
    # Test ProtBERT method (commented out as it requires large download)
    try:
        print("\n=== Testing ProtBERT Method ===")
        features_protbert = get_features(df_sample, method='protbert', batch_size=2)
        print(f"ProtBERT features shape: {features_protbert.shape}")
        print("First few columns:", list(features_protbert.columns[:5]))
    except Exception as e:
        print(f"ProtBERT method failed: {e}")
