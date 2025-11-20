# Investor Presentation Charts & Visualizations

This directory contains data and specifications for charts to be generated for the investor presentation.

## Chart Specifications

### 1. ROI Comparison Chart

**Type**: Bar Chart  
**Data**: ROI by sport and threshold  
**File**: `roi_comparison_data.json`

**Data Points**:
- NHL Meta-Ensemble ≥65%: 32.5%
- NHL Meta-Ensemble ≥60%: 26.5%
- NHL GBM ≥60%: 24.4%
- NHL Meta-Ensemble ≥55%: 21.5%
- NFL QB Edge: 27.3%
- NBA Elite Team: 7.6%

### 2. Win Rate Distribution

**Type**: Histogram/Box Plot  
**Data**: Win rate by threshold  
**File**: `win_rate_distribution_data.json`

**Data Points**:
- NHL Meta-Ensemble ≥65%: 69.4% (mean), 4.8% (std dev)
- NHL Meta-Ensemble ≥60%: 66.3% (mean), 2.3% (std dev)
- NFL QB Edge: 66.7%
- NBA Elite Team: 54.5%

### 3. Volume vs ROI Tradeoff

**Type**: Scatter Plot  
**Data**: Bets per season vs ROI  
**File**: `volume_roi_tradeoff_data.json`

**Data Points**:
- NHL ≥65%: 85 bets, 32.5% ROI
- NHL ≥60%: 577 bets, 24.4% ROI
- NHL ≥55%: 1,356 bets, 21.5% ROI
- NFL: 20 bets, 27.3% ROI
- NBA: 11 bets, 7.6% ROI

### 4. Training vs Production Performance

**Type**: Comparison Bar Chart  
**Data**: Training vs Production win rates and ROI  
**File**: `training_production_comparison_data.json`

**Data Points**:
- NHL Training (≥65%): 95.8% win, 82.9% ROI
- NHL Production (≥65%): 69.4% win, 32.5% ROI
- NFL Training: 61.5% win, 17.5% ROI
- NFL Production: 66.7% win, 27.3% ROI

### 5. Portfolio Profit Projections

**Type**: Stacked Bar Chart  
**Data**: Expected profit by scenario  
**File**: `portfolio_profit_projections_data.json`

**Data Points**:
- Conservative: $3,393/season
- Moderate: $14,625/season
- Aggressive: $29,700/season

### 6. Market Efficiency Spectrum

**Type**: Horizontal Bar Chart  
**Data**: ROI by sport (efficiency ranking)  
**File**: `market_efficiency_spectrum_data.json`

**Data Points**:
- NHL: 32.5% ROI (Least Efficient)
- NFL: 27.3% ROI (Moderately Efficient)
- NBA: 7.6% ROI (Most Efficient)

### 7. Statistical Significance Visualization

**Type**: Forest Plot  
**Data**: Win rates with confidence intervals  
**File**: `statistical_significance_data.json`

**Data Points**:
- NHL ≥65%: 69.4% [59.2%, 78.5%]
- NHL ≥60%: 66.3% [61.6%, 70.8%]
- NFL: 66.7% [35.9%, 90.1%]
- NBA: 54.5% [39.8%, 68.7%]

## Chart Generation

Charts can be generated using:
- Python (matplotlib, seaborn, plotly)
- R (ggplot2)
- JavaScript (D3.js, Chart.js)
- Excel/Google Sheets

## Data Files

All data files are in JSON format for easy integration with visualization tools.

