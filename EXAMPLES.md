# Single-File Hybrid Ensemble - Usage Examples

## Quick Start Examples

### 1. Basic Bitcoin Analysis
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast
```

**Output:**
```
--- הורדת נתונים: BTC-USD period=5y interval=1d ---
[INFO] Using local CSV data...
[INFO] Loaded 2713 rows from CSV

===== תוצאות מודלים =====
ARIMA_GARCH : 0.3078
Fundamental : 0.6500
GRU         : 0.2019
LSTM        : 0.7102
MonteCarlo  : 0.5600
Prophet     : 1.0000
Technical   : 0.4500
XGBoost     : 0.3008
-------------------------
Final Ensemble Score: 0.5236
Recommendation: Hold
Timestamp: 2025-09-20 20:17:42 UTC

(אין באמור ייעוץ השקעות / Educational Only)
```

### 2. Time-Horizon Specific Analysis

#### Short-term prediction (30 days)
Focuses on ARIMA-GARCH, LSTM, and HMM models for detecting short-term volatility and market state changes:
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast --time_horizon 30
```

**Output:**
```
===== תוצאות מודלים =====
Time Horizon: 30 days (Focus: ARIMA-GARCH, LSTM, HMM)
-------------------------
ARIMA_GARCH : 0.3078 (weight: 0.35)
HMM         : 0.6700 (weight: 0.25)
LSTM        : 0.7891 (weight: 0.30)
Technical   : 0.4500 (weight: 0.10)
[Other models]: (not used)
-------------------------
Final Ensemble Score: 0.5570
Recommendation: Hold
```

#### Medium-term prediction (60 days)
Focuses on LSTM, XGBoost, and ARIMA-GARCH for capturing medium-term dependencies:
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast --time_horizon 60
```

#### Longer-short term prediction (90 days)
Focuses on Factor Models, LSTM, and XGBoost for trend prediction with multiple factors:
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast --time_horizon 90
```

### 3. Custom Model Weights
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast --weights '{"XGBoost":0.25,"LSTM":0.25,"Technical":0.25,"Fundamental":0.25}'
```

### 4. Debug Mode Analysis
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast --debug
```

**Sample Debug Output:**
```
[DEBUG] Normalized weights sum -> 1.0000
[DEBUG] Adding technical indicators...
[DEBUG] Feature cols used: ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Volatility_20']
[DEBUG] FAST mode: reducing epochs & sims.
[DEBUG][XGBoost] Last prob: 0.3008
[DEBUG][LSTM] pred=42189.7507, current=40126.4297, score=0.4114
[DEBUG][GRU] pred=41320.9145, current=40126.4297, score=0.2381
[DEBUG][Prophet] forecast=61512.3986 last=40126.4297 score=1.0000
[DEBUG][ARIMA] pmdarima not installed -> 0.5
[DEBUG][GARCH] vol=0.035372 score=0.1157
[DEBUG][MonteCarlo] mu=0.001792 sigma=0.039226 score=0.5100
[DEBUG][Technical] rsi=44.55 macd=155.67261 sig=334.24094 score=0.4500
[DEBUG][Fundamental] score=0.6500 (override=None)
[DEBUG] Active weight sum used: 1.0000
[DEBUG] Final Score Raw: 0.477424
```

### 4. JSON Export
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast --save_json analysis.json
```

**JSON Output Structure:**
```json
{
  "ticker": "BTC-USD",
  "timestamp": "2025-09-20 20:15:59 UTC",
  "scores": {
    "XGBoost": 0.3008047640323639,
    "LSTM": 0.8737904784020152,
    "GRU": 0.4050080014707301,
    "Prophet": 1.0,
    "ARIMA_GARCH": 0.3078438672919639,
    "MonteCarlo": 0.555,
    "Technical": 0.45,
    "Fundamental": 0.65
  },
  "final_score": 0.5869546354366817,
  "decision": "Hold",
  "weights_used": {
    "XGBoost": 0.2,
    "LSTM": 0.2,
    "GRU": 0.15,
    "Prophet": 0.15,
    "ARIMA_GARCH": 0.1,
    "MonteCarlo": 0.1,
    "Technical": 0.05,
    "Fundamental": 0.05
  }
}
```

### 5. Fundamental Override
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast --fundamental_override 0.8
```

### 6. Technical Analysis Focused
```bash
python single_ensemble.py --ticker BTC-USD --use_csv --fast --weights '{"Technical":0.5,"XGBoost":0.2,"LSTM":0.2,"Fundamental":0.1}'
```

## Model Descriptions

### Default Model Weights (when no time horizon is specified)
| Model | Weight | Description |
|-------|--------|-------------|
| **XGBoost** | 15% | Binary classification predicting price direction |
| **LSTM** | 15% | Deep neural network for sequence prediction |
| **GRU** | 10% | Gated recurrent unit for time series |
| **Prophet** | 10% | Facebook's time series forecasting |
| **ARIMA+GARCH** | 10% | Statistical time series + volatility modeling |
| **Monte Carlo** | 10% | Stochastic simulation of price movements |
| **Technical** | 15% | RSI, MACD indicators with rule-based logic |
| **Fundamental** | 15% | Configurable fundamental analysis score |

### Time-Horizon Specific Models
| Model | Description | Best For |
|-------|-------------|----------|
| **HMM** | Hidden Markov Model for detecting market state changes | 30-day short-term predictions |
| **Factor Models** | Multi-factor regression model (Fama-French style) | 90-day longer-term trend prediction |

### Time Horizon Configurations
| Horizon | Focus Models | Rationale |
|---------|--------------|-----------|
| **30 days** | ARIMA-GARCH (35%), LSTM (30%), HMM (25%) | Volatility modeling and short-term pattern detection |
| **60 days** | LSTM (40%), XGBoost (25%), ARIMA-GARCH (20%) | Medium-term dependencies with external factors |
| **90 days** | Factor Models (35%), LSTM (30%), XGBoost (20%) | Long-term trends with fundamental factors |

### Interpreting Time-Horizon Results

**30-Day Results**: Focus on immediate market volatility and state changes. The HMM model helps detect sudden market regime shifts, while ARIMA-GARCH captures short-term volatility patterns.

**60-Day Results**: Balanced approach combining pattern recognition (LSTM) with feature-based predictions (XGBoost). Good for medium-term investment decisions.

**90-Day Results**: Emphasizes fundamental factors and longer-term trends. Factor Models analyze multiple market influences while LSTM captures long-term dependencies.

## Investment Recommendation Scale

| Score Range | Recommendation | Action |
|-------------|----------------|---------|
| 0.70-1.00 | **Strong Buy** | 💚 High confidence buy signal |
| 0.60-0.69 | **Buy** | 🟢 Moderate buy signal |
| 0.50-0.59 | **Hold** | 🟡 Neutral, hold current position |
| 0.40-0.49 | **Sell** | 🟠 Moderate sell signal |
| 0.00-0.39 | **Strong Sell** | 🔴 High confidence sell signal |

## Command-Line Options

```
--ticker TICKER                    Stock/crypto symbol (default: XLC)
--period PERIOD                    Time period: 1y, 2y, 5y, max (default: 5y)
--interval INTERVAL                Data interval: 1d, 1h, 1wk (default: 1d)
--lookback LOOKBACK               LSTM/GRU sequence length (default: 80)
--epochs_lstm EPOCHS_LSTM         LSTM training epochs (default: 6)
--epochs_gru EPOCHS_GRU           GRU training epochs (default: 6)
--fast                            Quick analysis mode (fewer epochs/simulations)
--debug                           Detailed debug output for each model
--use_csv                         Use local BTC-USD.csv instead of downloading
--weights WEIGHTS                 Custom JSON model weights
--save_json SAVE_JSON             Export results to JSON file
--fundamental_override OVERRIDE   Override fundamental score (0-1)
--time_horizon {30,60,90}         Time horizon for prediction focus:
                                    30 = Short-term (ARIMA-GARCH, LSTM, HMM)
                                    60 = Medium-term (LSTM, XGBoost, ARIMA-GARCH)
                                    90 = Longer-short term (Factor Models, LSTM, XGBoost)
```

## Testing

Run the comprehensive test suite:
```bash
python test_ensemble.py
```

Expected output:
```
🧪 Testing Single-File Hybrid Ensemble System
==================================================
✅ Basic Analysis: PASSED
✅ Custom Weights: PASSED
✅ JSON Export: PASSED
✅ Debug Mode: PASSED
✅ Fundamental Override: PASSED
==================================================
🏁 Test Results: 5/5 tests passed
🎉 All tests passed! The ensemble system is working correctly.
```

## Educational Disclaimer

⚠️ **This system is for educational and research purposes only.**
- Results should not be used for actual investment decisions
- Past performance does not guarantee future results
- Always consult qualified financial advisors
- Use at your own risk

**אזהרה: מערכת זו מיועדת למטרות חינוכיות ומחקריות בלבד. אין לראות בתוצאות ייעוץ השקעות.**