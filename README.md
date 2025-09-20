# Bitcoin Price Predictor & NVDA Hybrid Stock Analysis

This repository contains both a Bitcoin price predictor using LSTM and a comprehensive NVDA stock analysis system using multiple AI/ML models.

## Features

### Bitcoin Analysis (Original)
- Bitcoin price prediction using Long Short-Term Memory (LSTM) neural networks
- Technical analysis with historical price data
- Visualization of predictions vs actual prices

### NVDA Hybrid Analysis System (New)
A comprehensive stock analysis system for NVDA that combines multiple models:

- **XGBoost**: Gradient boosting for feature-based predictions
- **LSTM**: Long Short-Term Memory neural networks for sequence learning
- **GRU**: Gated Recurrent Units for time series analysis
- **ARIMA**: Auto-Regressive Integrated Moving Average for time series forecasting
- **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands
- **Sentiment Analysis**: News and social media sentiment (mockup)
- **Monte Carlo Simulation**: Statistical simulation for price distribution
- **Ensemble Model**: Weighted combination of all models for final recommendation

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

### Required Libraries

For Bitcoin analysis:
- pandas, numpy, math, matplotlib.pyplot
- scikit-learn (sklearn)
- tensorflow, keras
- plotly

For NVDA hybrid analysis:
- All of the above plus:
- xgboost
- yfinance  
- textblob
- transformers
- torch
- statsmodels
- ta (technical analysis)
- seaborn

## Usage

### Bitcoin Analysis
```python
python PricePredictor.py
```

### NVDA Hybrid Analysis
```python
python nvda_hybrid_predictor.py
```

### Combined Analysis
The main `PricePredictor.py` now includes both Bitcoin LSTM analysis and NVDA hybrid analysis.

### Test System
```python
python nvda_lite_test.py  # Lightweight test without dependencies
python test_nvda_system.py  # Full system test (requires all packages)
```

## Model Weights

The ensemble system uses the following weights for combining predictions:

- XGBoost: 20%
- LSTM: 18% 
- GRU: 15%
- ARIMA: 12%
- Technical Analysis: 15%
- Sentiment Analysis: 10%
- Monte Carlo: 10%

## Output

The system provides:
- Individual model predictions
- Technical indicator analysis
- Ensemble recommendation (BUY/SELL/HOLD)
- Confidence levels
- Comprehensive analysis report
- Visualizations and charts

## Installation Notes

### For Windows:
```bash
pip install -U scikit-learn
pip install tensorflow
```

### For Mac:
```bash
pip install -U numpy scipy scikit-learn
pip install tensorflow
```

### For TensorFlow:
Requires Python 3.7 or better:
```bash
python -m pip install --upgrade pip
pip install tensorflow
```

## Architecture

The system implements a hybrid ensemble approach combining:
- **Statistical Models**: ARIMA, Monte Carlo
- **Machine Learning**: XGBoost, Random Forest
- **Deep Learning**: LSTM, GRU, Transformers
- **Technical Analysis**: Traditional trading indicators
- **Sentiment Analysis**: News and social media sentiment
- **Ensemble Learning**: Weighted combination with confidence scoring

## Files

- `PricePredictor.py` - Original Bitcoin predictor + integrated NVDA analysis
- `nvda_hybrid_predictor.py` - Full NVDA hybrid analysis system
- `nvda_lite_test.py` - Lightweight demo version (no dependencies)
- `test_nvda_system.py` - System testing script
- `requirements.txt` - Required Python packages
- `BTC-USD.csv` - Bitcoin historical data

## Performance

The system provides performance metrics for each model:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- Variance explained
- Confidence intervals

## Future Enhancements

- Real-time data integration
- More sophisticated sentiment analysis
- Additional technical indicators
- Portfolio optimization
- Risk management features
- Web interface