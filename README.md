# 🤖 Bitcoin Price Predictor - AI Hybrid Stock Analysis System

This repository contains a comprehensive Bitcoin price prediction system with both a Python LSTM model and an advanced React-based web interface featuring 10 AI models for hybrid stock analysis.

## 🌟 Features

### New: Single-File Hybrid Ensemble System (single_ensemble.py)
- **8 AI Models in One**: XGBoost, LSTM, GRU, Prophet, ARIMA+GARCH, Monte Carlo, Technical Analysis, Fundamental Analysis  
- **Command-Line Interface** with extensive configuration options
- **Weighted Ensemble Scoring** (0-1 scale) with customizable model weights
- **Multi-Ticker Support** - analyze any stock/crypto, not just Bitcoin
- **Fast Mode** for quick analysis or full mode for detailed predictions
- **Debug Mode** with detailed model execution insights
- **JSON Export** for programmatic integration
- **Hebrew Documentation** with educational disclaimers
- **Graceful Degradation** - works even if some packages are missing

### Web Interface (React App)
- **AI Hybrid Analysis System** with 10 different ML models
- **Interactive Dashboard** with real-time charts and visualizations
- **CSV File Upload** support for custom datasets
- **Multi-language Support** (Hebrew/English)
- **Responsive Design** with gradient animations
- **Technical Analysis** with SMA, RSI indicators
- **Ensemble Predictions** with confidence scoring

### Python LSTM Model (Original)
- Long Short-Term Memory neural network for Bitcoin price prediction
- Historical data analysis and visualization
- Model performance evaluation metrics

## 🚀 Getting Started

### Single-File Hybrid Ensemble (Recommended)

1. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```

2. **Basic Usage**
```bash
# Analyze Bitcoin using local CSV data (fast mode)
python single_ensemble.py --ticker BTC-USD --use_csv --fast

# Analyze any stock with debug output
python single_ensemble.py --ticker AAPL --debug

# Custom model weights with JSON export
python single_ensemble.py --ticker NVDA --weights '{"XGBoost":0.3,"LSTM":0.2,"Technical":0.2,"Fundamental":0.3}' --save_json analysis.json

# Override fundamental analysis score
python single_ensemble.py --ticker MSFT --fundamental_override 0.75 --debug
```

3. **Command-Line Options**
```
--ticker          Stock/crypto symbol (default: XLC)
--period          Time period: 1y, 2y, 5y, max (default: 5y)  
--interval        Data interval: 1d, 1h, 1wk (default: 1d)
--fast            Quick analysis mode (fewer epochs/simulations)
--debug           Detailed debug output for each model
--use_csv         Use local BTC-USD.csv instead of downloading
--weights         Custom JSON model weights
--save_json       Export results to JSON file
--fundamental_override  Override fundamental score (0-1)
--lookback        LSTM/GRU sequence length (default: 80)
--epochs_lstm     LSTM training epochs (default: 6)  
--epochs_gru      GRU training epochs (default: 6)
```

### Model Descriptions

1. **XGBoost** (15% weight): Binary classification predicting price direction
2. **LSTM** (15% weight): Deep neural network for sequence prediction  
3. **GRU** (10% weight): Gated recurrent unit for time series
4. **Prophet** (10% weight): Facebook's time series forecasting
5. **ARIMA+GARCH** (10% weight): Statistical time series + volatility modeling
6. **Monte Carlo** (10% weight): Stochastic simulation of price movements
7. **Technical Analysis** (15% weight): RSI, MACD indicators with rule-based logic
8. **Fundamental** (15% weight): Configurable fundamental analysis score

### Investment Recommendations Scale
- **0.70-1.00**: Strong Buy 💚
- **0.60-0.69**: Buy 🟢  
- **0.50-0.59**: Hold 🟡
- **0.40-0.49**: Sell 🟠
- **0.00-0.39**: Strong Sell 🔴

### Web Interface Setup

1. **Install Dependencies**
```bash
npm install
```

2. **Run Development Server**
```bash
npm start
```
The application will open at `http://localhost:3000`

3. **Build for Production**
```bash
npm run build
```

### Python Model Setup (Original)

**Required Libraries:**
- pandas, numpy, math, matplotlib.pyplot
- scikit-learn: `pip install -U scikit-learn`
- TensorFlow: `pip install tensorflow` (requires Python 3.7+)
- Plotly: `pip install plotly`

**Usage:**
```bash
python PricePredictor.py
```

## 📊 Example Analysis Results

```
===== תוצאות מודלים =====
ARIMA_GARCH : 0.3078
Fundamental : 0.6500
GRU         : 0.4050
LSTM        : 0.8738
MonteCarlo  : 0.5550
Prophet     : 1.0000
Technical   : 0.4500
XGBoost     : 0.3008
-------------------------
Final Ensemble Score: 0.5870
Recommendation: Hold
Timestamp: 2025-09-20 20:15:59 UTC

(אין באמור ייעוץ השקעות / Educational Only)
```

## 📁 File Structure

```
bitcoin-price-predictor/
├── single_ensemble.py      # 🆕 Hybrid ensemble system (main)
├── requirements.txt        # Python dependencies
├── btc_analysis.json      # Example output file
├── public/                # Web app public files
├── src/                   # React source code
│   ├── components/        # React components
│   └── index.js          # Main app entry point
├── build/                # Production build
├── PricePredictor.py     # Original Python LSTM model
├── BTC-USD.csv          # Bitcoin historical data
├── package.json         # Node.js dependencies
└── README.md           # This file
```

## 🎯 CSV Data Format

The system accepts CSV files with the following columns:
- `Date` - Date in YYYY-MM-DD format
- `Close` - Closing price
- `Open` - Opening price (optional)
- `High` - Highest price (optional)
- `Low` - Lowest price (optional)
- `Volume` - Trading volume (optional)

## 🛠️ Technology Stack

**Hybrid Ensemble:**
- Python 3.7+ with TensorFlow, XGBoost, Prophet
- yfinance for real-time data
- Statistical models: ARIMA, GARCH
- Technical indicators: RSI, MACD, SMA

**Frontend:**
- React 18
- Recharts for data visualization
- Lucide React for icons
- Papa Parse for CSV parsing
- CSS3 with gradients and animations

**Backend/Analysis:**
- Python with TensorFlow/Keras
- scikit-learn for metrics
- Pandas for data processing
- Plotly for visualization

## ⚠️ Educational Disclaimer

**This system is for educational and research purposes only. It is NOT financial advice.**
- Results should not be used for actual investment decisions
- Past performance does not guarantee future results
- Always consult qualified financial advisors
- Use at your own risk

**אזהרה: מערכת זו מיועדת למטרות חינוכיות ומחקריות בלבד. אין לראות בתוצאות ייעוץ השקעות.**

## 📈 Screenshots

The web interface provides:
- Interactive price charts
- Real-time analysis progress
- Comprehensive model results
- Professional gradient design
- Multi-language support