"""
Data preprocessing utilities for peptide analysis.
Handles data loading, cleaning, and preparation.
"""

import pandas as pd
import os


def load_and_process_mic_data(filepath):
    """
    Load and process MIC data from Excel file.
    
    Parameters:
    -----------
    filepath : str
        Path to the Excel file containing MIC data
        
    Returns:
    --------
    tuple
        (sequences_df, labels) - DataFrame with sequences and labels Series
    """
    # Load data
    df = pd.read_excel(filepath)
    df.sequence = df.sequence.str.strip()
    
    # Select relevant columns and group by sequence
    df = df[['sequence', 'active']]
    df = df.groupby('sequence').mean()
    df.reset_index(inplace=True)
    
    # Create labels
    df['label'] = df['active'].apply(lambda x: 1 if x >= 0.5 else (0 if 0 <= x < 0.5 else -1))
    
    # Filter valid labels
    df = df[df.label >= 0]
    df.reset_index(drop=True, inplace=True)
    
    # Keep only necessary columns
    df = df[['sequence', 'label']]
    
    # Remove duplicates
    df.drop_duplicates(subset=['sequence'], inplace=True)
    
    print(f"Processed {len(df)} unique sequences")
    
    return df, df['label']


def save_data(data, filepath):
    """
    Save data to pickle file.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Data to save
    filepath : str
        Path to save the file
    """
    data.to_pickle(filepath)
    print(f"Data saved to {filepath}")


def load_data(filepath):
    """
    Load data from pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to the pickle file
        
    Returns:
    --------
    pd.DataFrame or pd.Series
        Loaded data
    """
    data = pd.read_pickle(filepath)
    print(f"Data loaded from {filepath}")
    return data
