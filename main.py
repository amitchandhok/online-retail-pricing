import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from src.data import make_dataset, process_data
from src.features import build_features
from src.models import train_model
from src.vizualization import vizualize

def main():
    # Load data
    print("Loading data...")
    df, df_promo = make_dataset.load_raw_data()
    
    # Process data
    print("Processing data...")
    df = process_data.correct_outliers(df)
    df = process_data.clean_data(df)
    
    # Feature engineering
    print("Creating features...")
    train_df, test_df = build_features.time_based_split(df)
    train_df, test_df = build_features.create_product_clusters(train_df, test_df)
    train_df = build_features.engineer_features(train_df)
    full_feature_columns = train_df.columns
    test_df = build_features.engineer_features(test_df, full_columns=full_feature_columns)
    
    # Visualization
    print("Creating visualizations...")
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    vizualize.plot_price_quantity(df)
    vizualize.plot_top_skus(df)
    vizualize.plot_revenue_concentration(df)
    
    # Modeling
    print("Training models...")
    results_df = train_model.train_sku_models(train_df, test_df)
    Path("reports/results").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("reports/results/model_results.csv", index=False)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()