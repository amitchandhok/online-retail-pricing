import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from typing import Dict, List

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate model performance"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': sm.OLS(y_true, sm.add_constant(y_pred)).fit().rsquared
    }

def train_sku_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Train individual models for each SKU"""
    sku_list = train_df['StockCode'].unique()
    results = []
    
    feature_cols = [
        'LogPrice', 'LogQuantityLag1', 'LogQuantityLag2', 'LogQuantityLag3', 'LogQuantityLag4', 
        'AvgQuantityRolling2', 'AvgQuantityRolling3', 'AvgQuantityRolling4', 'AvgQuantityRolling5',
        'IsPromo', 'PriceChange', 'PriceChangeLag', 'QuarterSin', 'QuarterCos', 'MonthSin', 'MonthCos'
    ] + [col for col in train_df.columns if col.startswith(('Category_','Day_','Month_', 'Quarter_'))]

    for sku in sku_list:
        sku_train = train_df[train_df['StockCode'] == sku]
        sku_test = test_df[test_df['StockCode'] == sku]
        
        X_train = sm.add_constant(sku_train[feature_cols], has_constant='add')
        X_test = sm.add_constant(sku_test[feature_cols], has_constant= 'add')
        y_train = sku_train['LogQuantity']
        y_test = sku_test['LogQuantity']

        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)
        
        test_metrics = evaluate_model(np.exp(y_test), np.exp(y_pred))
        train_metrics = evaluate_model(np.exp(y_train), np.exp(model.predict(X_train)))

        conf_int = model.conf_int().loc['LogPrice']
        
        results.append({
            'StockCode': sku,
            'Elasticity': model.params.get('LogPrice', np.nan),
            'ElasticityPval': model.pvalues.get('LogPrice', np.nan),
            'ElasticityCILower': conf_int[0],
            'ElasticityCIUpper': conf_int[1],
            'ModelFPval': model.f_pvalue,
            'TrainMAE': train_metrics['MAE'],
            'TestMAE': test_metrics['MAE'],
            'TrainR2': train_metrics['R2'],
            'TestR2': test_metrics['R2'],
            'TrainObs': len(sku_train),
            'TestObs': len(sku_test)
        })

    return pd.DataFrame(results)