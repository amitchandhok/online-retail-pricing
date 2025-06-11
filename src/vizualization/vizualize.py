import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter

def plot_price_quantity(df: pd.DataFrame, output_dir: str = 'reports/figures'):
    """Plot price vs quantity relationships"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='UnitPrice', y='Quantity', alpha=0.4)
    plt.title('Price vs Quantity (All SKUs)')
    plt.xlabel('Unit Price')
    plt.ylabel('Quantity')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    df_log = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    df_log['log_Quantity'] = np.log(df_log['Quantity'])
    df_log['log_UnitPrice'] = np.log(df_log['UnitPrice'])
    
    sns.scatterplot(data=df_log, x='log_UnitPrice', y='log_Quantity', alpha=0.4)
    sns.regplot(data=df_log, x='log_UnitPrice', y='log_Quantity',
                scatter=False, color='red', label='Linear Fit')
    plt.title('Log-Log Price vs Quantity')
    plt.xlabel('Log(Unit Price)')
    plt.ylabel('Log(Quantity)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/price_quantity_relationship.png')
    plt.close()

def plot_top_skus(df: pd.DataFrame, output_dir: str = 'reports/figures'):
    """Plot top SKUs by revenue"""
    top_skus = df.groupby(['StockCode','Description'])['TotalRevenue'].sum().nlargest(20)
    
    plt.figure(figsize=(12,8))
    top_skus.sort_values().plot(kind='barh', color="#135AE8D9")
    plt.title('Top 20 SKUs by Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel('SKU & Description')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_skus_by_revenue.png')
    plt.close()

def plot_revenue_concentration(df: pd.DataFrame, output_dir: str = 'reports/figures'):
    """Plot revenue concentration"""
    revenue_by_sku = df.groupby('StockCode')['TotalRevenue'].sum().sort_values(ascending=False)
    top_4 = revenue_by_sku.head(4)
    pct_of_total = top_4.sum() / revenue_by_sku.sum() * 100
    sku_names = revenue_by_sku.index.astype(str)  
    cumulative_pct = revenue_by_sku.cumsum() / revenue_by_sku.sum() * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(sku_names)), revenue_by_sku, color="#135AE8D9")
    ax.set_xticks(range(len(sku_names)))  
    ax.set_xticklabels(sku_names, rotation=45, ha='right', rotation_mode='anchor')

    for i in range(4):
        bars[i].set_color("#F8D700")

    ax2 = ax.twinx()
    ax2.plot(range(len(sku_names)), cumulative_pct, color='red', marker='D', ms=5)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    ax2.axhline(y=pct_of_total, color='red', linestyle='--')
    ax2.text(x=2, y=pct_of_total+2, s=f'Top 4 SKUs = {pct_of_total:.0f}% Revenue', 
             color='red', weight='bold')

    ax.set_title("Revenue Concentration", pad=20) 
    ax.set_xlabel("SKU")
    ax.set_ylabel("Total Revenue ($)", color='steelblue')
    ax2.set_ylabel("Cumulative % of Revenue", color='red')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/revenue_concentration.png')
    plt.close()

def sku_correlation(df: pd.DataFrame, output_dir: str = 'reports/figures'):
    """Plot correlation between SKUs"""
    # Calculate pairwise correlation of purchase co-occurrence
    pivot = df.pivot_table(index='InvoiceNo', columns='StockCode', 
                        values='Quantity', fill_value=0)
    corr_matrix = pivot.corr()

    # Cluster and plot
    sns.clustermap(corr_matrix, cmap='coolwarm', center=0, figsize=(12,10))
    plt.title('SKU Purchase Correlations')
    plt.savefig(f'{output_dir}/sku_correlation.png')
    plt.close()