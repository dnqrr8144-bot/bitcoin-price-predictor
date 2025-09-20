# 🤖 Bitcoin Price Predictor - AI Hybrid Stock Analysis System

This repository contains a comprehensive Bitcoin price prediction system with both a Python LSTM model and an advanced React-based web interface featuring 10 AI models for hybrid stock analysis.

## 🌟 Features

### Web Interface (New!)
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

### Python Model Setup

**Required Libraries:**
- pandas, numpy, math, matplotlib.pyplot
- scikit-learn: `pip install -U scikit-learn`
- TensorFlow: `pip install tensorflow` (requires Python 3.7+)
- Plotly: `pip install plotly`

**Usage:**
```bash
python PricePredictor.py
```

## 📊 Web Interface Usage

1. **Upload Data**: Click "העלה קובץ CSV" to upload your CSV file (or use the default NVDA mock data)
2. **Run Analysis**: Click "🚀 הפעל אנליזה היברידית מתקדמת" to start the AI analysis
3. **View Results**: Get comprehensive analysis including:
   - Final investment score (0-100)
   - Buy/Sell/Hold recommendations
   - Model confidence levels
   - Target prices and potential returns
   - Individual model predictions

## 🤖 AI Models Included

1. **XGBoost Ensemble** (15% weight)
2. **LSTM Deep Neural Network** (14% weight)  
3. **Transformer Model** (13% weight)
4. **GRU Network** (11% weight)
5. **Neuroadaptive Technical Analysis** (12% weight)
6. **Quantum-Enhanced TA** (9% weight)
7. **Monte Carlo Simulation** (8% weight)
8. **GARCH Volatility Model** (7% weight)
9. **ARIMA Forecasting** (6% weight)
10. **Sentiment Analysis AI** (5% weight)

## 📁 File Structure

```
bitcoin-price-predictor/
├── public/                 # Web app public files
├── src/                    # React source code
│   ├── components/         # React components
│   └── index.js           # Main app entry point
├── build/                 # Production build
├── PricePredictor.py      # Original Python LSTM model
├── BTC-USD.csv           # Bitcoin historical data
├── package.json          # Node.js dependencies
└── README.md             # This file
```

## 🎯 CSV Data Format

The system accepts CSV files with the following columns:
- `Date` - Date in YYYY-MM-DD format
- `Close` - Closing price
- `Open` - Opening price
- `High` - Highest price
- `Low` - Lowest price  
- `Volume` - Trading volume

## 🛠️ Technology Stack

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

## 📈 Screenshots

The web interface provides:
- Interactive price charts
- Real-time analysis progress
- Comprehensive model results
- Professional gradient design
- Multi-language support