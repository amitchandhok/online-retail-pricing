import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(df):
    """Perform data cleaning steps"""
    # Filter to see product SKU only
    df['StockCode'] = df['StockCode'].astype(str).str.upper()
    df = df[df['StockCode'].astype(str).str.match(r'^([0-9]|D)[0-9C]')]

    # Calculate total revenue and filter top SKUs in UK
    df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']
    top20 = df[df['Country'] == 'United Kingdom'].groupby('StockCode')['TotalRevenue'].sum().nlargest(20).index
    top20_quantity = df[df['Country'] == 'United Kingdom'].groupby('StockCode')['Quantity'].sum().nlargest(10).index
    selected_skus = list(set(top20) | set(top20_quantity))
    df = df[(df['Country'] == 'United Kingdom') & (df['StockCode'].isin(selected_skus))]

    # Standardize descriptions
    desc_mapping = df.groupby('StockCode')['Description'].apply(lambda x: x.mode()[0]).to_dict()
    df['Description'] = df['StockCode'].map(desc_mapping)
    
    # Clean InvoiceNo and aggregate data
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.lstrip('C').astype(int)
    df = df.groupby(['InvoiceNo','StockCode','Description']).agg(
        {'Quantity': 'sum',
         'InvoiceDate': 'min',
         'UnitPrice': 'max',
         'TotalRevenue': 'sum'}
    ).sort_values(by='InvoiceDate').reset_index()

    # Convert InvoiceDate feature and filter positive values
    df['InvoiceDate'] = df['InvoiceDate'].dt.to_period('D')
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    return df

def correct_outliers(df):
    """Correct known outliers"""
    # Correct picnic basket pricing
    per_unit_price = (df[(df['StockCode'] == 22502) & 
                      (df['Description'] == 'PICNIC BASKET WICKER 60 PIECES')]['UnitPrice']/60).values[0]
    df.loc[(df['StockCode'] == 22502) & 
           (df['Description'] == 'PICNIC BASKET WICKER 60 PIECES'), 
           'UnitPrice'] = per_unit_price
    return df