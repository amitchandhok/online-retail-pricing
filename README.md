# Price Elasticity Analysis for Online Retail

## Project Overview

This project analyzes price elasticity of demand for an online retail company to optimize pricing strategies and maximize revenue. The analysis focuses on:

1. Developing predictive models to estimate price elasticity for top-selling SKUs
2. Evaluating the impact of a proposed promotional pricing plan
3. Providing data-driven recommendations for pricing optimization

## Dataset

The analysis uses two datasets:

1. **Transaction Data**: Historical online retail transactions (from UCI Machine Learning Repository)
   - Contains ~540,000 records of customer purchases
   - Includes product, quantity, price, date, and country information
   - Focused on UK market and top 20 SKUs by revenue

2. **Pricing Plan**: Proposed promotional pricing calendar for top 10 SKUs

## Key Steps

### 1. Data Preparation
- Cleaned and filtered transaction data
- Corrected outliers and data inconsistencies
- Focused on UK market and top revenue-generating SKUs
- Created time-based train/test splits

### 2. Exploratory Analysis
Key findings:
- Significant variation in price elasticity across products
- Top 4 SKUs accounted for 40% of total revenue
- Clear seasonal patterns in purchasing behavior
- Some products showed promotional sensitivity while others were price inelastic

### 3. Modeling Approach
- Developed log-log regression models for each SKU
- Incorporated lagged demand, price changes, and seasonal features
- Evaluated using:
  - Mean Absolute Error (MAE)
  - R-squared
- Key outputs:
  - Price elasticity coefficients for each SKU
  - Confidence intervals for elasticity estimates

### 4. Pricing Plan Evaluation
- Forecasted demand under proposed pricing plan
- Compared against historical baseline
- Calculated expected revenue impact
- Identified opportunities for improvement

## Key Findings

1. **Price Elasticity Insights**:
   - Elasticity coefficients ranged from -0.8 (elastic) to -0.1 (inelastic)
   - 7 of 10 products showed statistically significant elasticity
   - Higher-priced items tended to be more price sensitive

2. **Pricing Plan Impact**:
   - Overall revenue increase of 8.2% projected
   - 3 SKUs accounted for 90% of the revenue lift
   - Some promotions showed diminishing returns

3. **Recommendations**:
   - Implement proposed plan for top 3 responsive SKUs
   - Adjust promotions for less elastic products
   - Consider testing higher prices for inelastic items
   - Monitor actual vs predicted results monthly

## Repository Structure

```text
online-retail-pricing/
├── data/                 # Data files
│   ├── raw/              # Original data files
│   └── processed/        # Cleaned/processed data
├── notebooks/            # Jupyter notebooks
├── reports/              # Analysis outputs
│   ├── figures/          # Visualizations
│   └── results/          # Model outputs
├── src/                  # Python modules
├── main.py               # Main Python script
├── requirements.txt      # Requirements
└── README.md             # Project documentation
```

## How to Reproduce

1. Clone repository
2. Install requirements: `pip install -r requirements.txt`
3. Run analysis pipeline: `python main.py`

## Future Improvements

- Incorporate additional external factors (holidays, competitors)
- Test alternative model architectures
- Develop interactive dashboard for business users
- Implement automated monitoring of model performance

## Author

Amit Chandhok