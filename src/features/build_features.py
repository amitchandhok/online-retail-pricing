import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def create_product_clusters(train_df, test_df=None, n_clusters=5):
    """Cluster products based on descriptions"""
    tfidf = TfidfVectorizer(stop_words='english')
    desc_tfidf = tfidf.fit_transform(train_df['Description'].astype(str))
    
    svd = TruncatedSVD(n_components=4, random_state=42)
    desc_svd = svd.fit_transform(desc_tfidf)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    train_df['ProductCategory'] = kmeans.fit_predict(desc_svd)
    
    if test_df is not None:
        test_desc_tfidf = tfidf.transform(test_df['Description'].astype(str))
        test_desc_svd = svd.transform(test_desc_tfidf)
        test_df['ProductCategory'] = kmeans.predict(test_desc_svd)
    
    return train_df, test_df

def time_based_split(df, test_size=0.2):
    """Time-based train-test split by SKU"""
    train_list, test_list = [], []
    
    for sku in df['StockCode'].unique():
        sku_data = df[df['StockCode'] == sku].sort_values('InvoiceDate')
        cutoff_idx = int(len(sku_data) * (1 - test_size))
        
        train_list.append(sku_data.iloc[:cutoff_idx])
        test_list.append(sku_data.iloc[cutoff_idx:])
    
    return pd.concat(train_list), pd.concat(test_list)

def engineer_features(df, full_columns = None):
    """Create features for modeling"""
    # Group data
    df = df.groupby(['StockCode','InvoiceDate','ProductCategory']).agg(
        TotalQuantity=('Quantity', 'sum'),
        AvgPrice=('UnitPrice', 'mean')
    ).reset_index()

    # Log transformations
    df['LogQuantity'] = np.log(df['TotalQuantity'] + 1e-5)
    df['LogPrice'] = np.log(df['AvgPrice'] + 1e-5)
    
    # Lag features
    df = df.sort_values(['StockCode', 'InvoiceDate'])
    for lag in [1, 2, 3, 4]:
        df[f'QuantityLag{lag}'] = df.groupby('StockCode')['TotalQuantity'].shift(lag)
        df[f'LogQuantityLag{lag}'] = np.log(df[f'QuantityLag{lag}'] + 1e-5)

    # Time features
    df['Month'] = df['InvoiceDate'].dt.month
    df['Quarter'] = df['InvoiceDate'].dt.quarter
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_of_week
    
    # Cyclical features
    df['MonthSin'] = np.sin(2 * np.pi * df['Month']/12)
    df['MonthCos'] = np.cos(2 * np.pi * df['Month']/12)
    df['QuarterSin'] = np.sin(2 * np.pi * df['Quarter']/4)
    df['QuarterCos'] = np.cos(2 * np.pi * df['Quarter']/4)

    # Dummy variables
    category_dummies = pd.get_dummies(df['ProductCategory'], prefix='Category', drop_first=True).astype(int)
    day_dummies = pd.get_dummies(df['DayOfWeek'], prefix='Day').astype(int)
    month_dummies = pd.get_dummies(df['Month'], prefix='Month').astype(int)
    quarter_dummies = pd.get_dummies(df['Quarter'], prefix='Quarter').astype(int)

    # Concatenate features
    df = pd.concat([df, category_dummies, day_dummies, month_dummies, quarter_dummies], axis=1)
    
    # If full column set provided (i.e., we're processing test set)
    if full_columns is not None:
        missing_cols = [col for col in full_columns if col not in df.columns]
        for col in missing_cols:
            df[col] = 0
        df = df[full_columns]

    # Price dynamics
    df['PriceChange'] = df.groupby('StockCode')['AvgPrice'].pct_change()
    df['PriceChangeLag'] = df.groupby('StockCode')['PriceChange'].shift(1)
        
    # Promotional flags
    df['IsPromo'] = df.groupby('StockCode')['AvgPrice'].transform(
        lambda x: x < (x.rolling(3).mean() * 0.9)).astype(int)
    
    # Rolling demand stats
    for window in [2, 3, 4, 5]:
        df[f'AvgQuantityRolling{window}'] = df.groupby('StockCode')['TotalQuantity'].transform(
            lambda x: x.rolling(window).mean())
    
    # Clean up
    df.drop(columns=['Month', 'DayOfWeek', 'Quarter','ProductCategory'], inplace=True, errors='ignore')
    return df.dropna()