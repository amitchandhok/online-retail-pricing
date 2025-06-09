import pandas as pd
import numpy as np
from pathlib import Path

def load_raw_data(data_dir='data/raw'):
    """Load raw data files"""
    df = pd.read_excel(Path(data_dir) / 'Online Retail.xlsx')
    df_promo = pd.read_csv(Path(data_dir) / 'Promo Plan.csv')
    return df, df_promo

def save_processed_data(df, filename, data_dir='data/processed'):
    """Save processed data"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(data_dir) / filename, index=False)