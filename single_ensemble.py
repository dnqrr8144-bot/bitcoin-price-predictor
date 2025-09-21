#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-File Hybrid Ensemble (Detailed Edition)
----------------------------------------------
גרסה מורחבת עם EXPLAIN / DEBUG והערות בכל שלב.

מטרות:
1. קובץ יחיד להרצה פשוטה.
2. מודלים היברידיים (סטטיסטיים + למידת מכונה + עומק פשוט).
3. ניקוד אחיד 0–1 לכל מודל -> שקלול משקולות -> המלצה.

שימושים:
    python single_ensemble.py --ticker XLC
    python single_ensemble.py --ticker NVDA --fast
    python single_ensemble.py --ticker AAPL --weights '{"XGBoost":0.2,"LSTM":0.15,"GRU":0.1,"Prophet":0.1,"ARIMA_GARCH":0.1,"MonteCarlo":0.1,"Technical":0.15,"Fundamental":0.1}'
    python single_ensemble.py --ticker MSFT --fundamental_override 0.55 --debug

תלויות עיקריות (התקנה):
    pip install yfinance pandas numpy scikit-learn xgboost tensorflow pmdarima arch prophet

אם prophet נכשל – אפשר להסיר את החבילה, הקוד ייתן ציון נייטרלי (0.5) עבור הפונקציה שלו.

אזהרה:
אין לראות בתוצאות ייעוץ השקעות. הדגמה חינוכית בלבד.
"""
import warnings
warnings.filterwarnings("ignore")

# --- Standard Imports ---
import argparse
import json
from datetime import datetime, timezone

# --- Data / Math ---
import pandas as pd
import numpy as np
import yfinance as yf

# --- Time Series / Stats Models ---
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# --- Machine Learning ---
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Deep Learning (Keras) ---
try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    HAS_TENSORFLOW = True
except Exception:
    HAS_TENSORFLOW = False

# --- Prophet (Optional) ---
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False


# =========================
# Feature Engineering
# =========================
def add_technical_indicators(df: pd.DataFrame, debug=False) -> pd.DataFrame:
    """
    מוסיף אינדיקטורים טכניים בסיסיים לסדרת מחירים שהורדה מ-yfinance.
    דורש שהעמודות: Open, High, Low, Close, Volume קיימות.
    """
    df = df.copy()
    if debug:
        print("[DEBUG] Adding technical indicators...")

    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    # RSI (14) – חישוב בסיסי
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Rolling Volatility (20)
    df['Volatility_20'] = df['Close'].pct_change().rolling(20).std()

    # Drop NA after computing indicators (מאז שזה ממלא חלונות)
    df.dropna(inplace=True)
    return df

def build_feature_matrix(df: pd.DataFrame, debug=False):
    """
    מחזיר מטריצת פיצ'רים + רשימת שמות.
    """
    feature_list = [
        'Open','High','Low','Close','Volume',
        'SMA_20','SMA_50','RSI','MACD','MACD_Signal','Volatility_20'
    ]
    available = [f for f in feature_list if f in df.columns]
    if debug:
        print(f"[DEBUG] Feature cols used: {available}")
    return df[available].copy(), available


# =========================
# Model Implementations
# =========================
def model_xgboost(df, feature_cols, debug=False):
    """
    סיווג בינארי: האם close_{t+1} > close_t.
    """
    if not HAS_XGBOOST:
        if debug:
            print("[DEBUG][XGBoost] Not installed -> 0.5")
        return 0.5
        
    temp = df.copy()
    temp['Target'] = (temp['Close'].shift(-1) > temp['Close']).astype(int)
    temp.dropna(inplace=True)
    if temp['Target'].nunique() < 2:
        if debug:
            print("[DEBUG][XGBoost] Not enough class variance -> 0.5")
        return 0.5
    X = temp[feature_cols]
    y = temp['Target']
    # No shuffle -> לשימור סדר
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X.iloc[[-1]])[0][1]
    if debug:
        print(f"[DEBUG][XGBoost] Last prob: {prob:.4f}")
    return float(prob)

def _seq_builder(values, lookback):
    X, y = [], []
    for i in range(len(values) - lookback - 1):
        X.append(values[i:i+lookback])
        y.append(values[i+lookback])
    return np.array(X), np.array(y)

def _train_seq(close_series, lookback, epochs, cell='LSTM', debug=False):
    """
    אימון פשוט LSTM/GRU על מחירי סגירה בלבד.
    """
    if not HAS_TENSORFLOW:
        if debug:
            print(f"[DEBUG][{cell}] TensorFlow not installed -> 0.5")
        return 0.5
        
    if len(close_series) < lookback + 20:
        if debug:
            print(f"[DEBUG][{cell}] Not enough data length -> 0.5")
        return 0.5
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(close_series.values.reshape(-1,1))
    X, y = _seq_builder(arr.flatten(), lookback)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    if cell == 'LSTM':
        model.add(LSTM(48, return_sequences=True, input_shape=(lookback,1)))
        model.add(Dropout(0.25))
        model.add(LSTM(24))
    else:
        model.add(GRU(48, return_sequences=True, input_shape=(lookback,1)))
        model.add(Dropout(0.25))
        model.add(GRU(24))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    last = arr[-lookback:].reshape(1,lookback,1)
    pred_scaled = model.predict(last, verbose=0)[0][0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    current = close_series.values[-1]

    # Scoring heuristic
    if pred > current:
        score = min(1.0, (pred-current)/current * 8)
    else:
        score = max(0.0, 1 - (current-pred)/current * 8)

    if debug:
        print(f"[DEBUG][{cell}] pred={pred:.4f}, current={current:.4f}, score={score:.4f}")
    return float(score)

def model_lstm(df, lookback, epochs, debug=False):
    return _train_seq(df['Close'], lookback, epochs, 'LSTM', debug=debug)

def model_gru(df, lookback, epochs, debug=False):
    return _train_seq(df['Close'], lookback, epochs, 'GRU', debug=debug)

def model_prophet(df, debug=False):
    """
    Prophet תחזית יום קדימה.
    """
    if not HAS_PROPHET:
        if debug:
            print("[DEBUG][Prophet] Not installed -> 0.5")
        return 0.5
    s = df[['Close']].reset_index()
    s.columns = ['ds','y']
    try:
        m = Prophet(daily_seasonality=True, weekly_seasonality=True)
        m.fit(s)
        future = m.make_future_dataframe(periods=1)
        fc = m.predict(future).iloc[-1]['yhat']
    except Exception as e:
        if debug:
            print(f"[DEBUG][Prophet] Exception -> {e}")
        return 0.5
    last = df['Close'].iloc[-1]
    if fc > last:
        score = min(1.0,(fc-last)/last*8)
    else:
        score = max(0,(1-(last-fc)/last*8))
    if debug:
        print(f"[DEBUG][Prophet] forecast={fc:.4f} last={last:.4f} score={score:.4f}")
    return float(score)

def model_arima(df, debug=False):
    """
    Auto ARIMA – תחזית צעד אחד קדימה.
    """
    if not HAS_PMDARIMA:
        if debug:
            print("[DEBUG][ARIMA] pmdarima not installed -> 0.5")
        return 0.5
        
    try:
        model = pm.auto_arima(df['Close'], seasonal=False, suppress_warnings=True)
        fc = model.predict(1)[0]
    except Exception as e:
        if debug:
            print(f"[DEBUG][ARIMA] Exception -> {e}")
        return 0.5
    last = df['Close'].iloc[-1]
    if fc > last:
        score = min(1.0,(fc-last)/last*8)
    else:
        score = max(0,(1-(last-fc)/last*8))
    if debug:
        print(f"[DEBUG][ARIMA] forecast={fc:.4f} last={last:.4f} score={score:.4f}")
    return float(score)

def model_garch(df, debug=False):
    """
    GARCH(1,1) – הערכת תנודתיות. הופך ל-score באמצעות 1 - f(vol).
    """
    if not HAS_ARCH:
        if debug:
            print("[DEBUG][GARCH] arch not installed -> 0.5")
        return 0.5
        
    rets = df['Close'].pct_change().dropna()
    if len(rets) < 80:
        if debug:
            print("[DEBUG][GARCH] Not enough returns length -> 0.5")
        return 0.5
    try:
        am = arch_model(rets, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        f = res.forecast(horizon=1)
        vol = float((f.variance.values[-1,0])**0.5)
        score = 1 - min(1, vol * 25)
    except Exception as e:
        if debug:
            print(f"[DEBUG][GARCH] Exception -> {e}")
        return 0.5
    score = float(max(0, min(1, score)))
    if debug:
        print(f"[DEBUG][GARCH] vol={vol:.6f} score={score:.4f}")
    return score

def model_monte_carlo(df, sims=400, debug=False):
    """
    סימולציית log-return נורמלית, בודק אחוז עליות.
    """
    log_r = np.log(1 + df['Close'].pct_change().dropna())
    if len(log_r) < 50:
        if debug:
            print("[DEBUG][MonteCarlo] Not enough data -> 0.5")
        return 0.5
    mu = log_r.mean()
    sigma = log_r.std()
    last = df['Close'].iloc[-1]
    ups = 0
    for _ in range(sims):
        step = np.random.normal(mu, sigma)
        price = last * np.exp(step)
        if price > last:
            ups += 1
    score = float(ups/sims)
    if debug:
        print(f"[DEBUG][MonteCarlo] mu={mu:.6f} sigma={sigma:.6f} score={score:.4f}")
    return score

def model_technical(df, debug=False):
    """
    כלל החלטה פשוט משולב.
    """
    rsi = df['RSI'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    sig = df['MACD_Signal'].iloc[-1]
    score = 0.5
    if rsi < 30: score += 0.12
    if rsi > 70: score -= 0.12
    if macd > sig: score += 0.05
    else: score -= 0.05
    score = float(max(0,min(1,score)))
    if debug:
        print(f"[DEBUG][Technical] rsi={rsi:.2f} macd={macd:.5f} sig={sig:.5f} score={score:.4f}")
    return score

def model_fundamental_stub(override=None, debug=False):
    """
    פונדמנטלי סטטי (או override).
    """
    if override is not None:
        val = float(override)
    else:
        val = 0.65
    if debug:
        print(f"[DEBUG][Fundamental] score={val:.4f} (override={override})")
    return val

def model_hmm(df, debug=False):
    """
    Hidden Markov Model - מודל מארקוב סמוי לזיהוי מצבי שוק.
    מתאים במיוחד לטווח קצר (30 יום) לזיהוי שינויי מצבים חדים בשוק.
    """
    try:
        from sklearn.mixture import GaussianMixture
        
        # Prepare features - price changes and volatility
        returns = df['Close'].pct_change().dropna()
        volatility = returns.rolling(window=5).std().dropna()
        
        # Combine features for HMM
        features = np.column_stack([returns[-len(volatility):], volatility])
        features = features[~np.isnan(features).any(axis=1)]
        
        if len(features) < 10:
            if debug:
                print("[DEBUG][HMM] Insufficient data, returning neutral score")
            return 0.5
        
        # Fit Gaussian Mixture Model as HMM approximation with 2 states
        model = GaussianMixture(n_components=2, random_state=42)
        model.fit(features)
        
        # Predict current market state
        current_features = features[-1:] 
        state_probs = model.predict_proba(current_features)[0]
        
        # Get current state characteristics
        states = model.predict(features)
        recent_state = states[-1]
        
        # Analyze if current state suggests upward movement
        # State with higher mean return gets higher score
        state_returns = []
        for state in range(2):
            state_mask = states == state
            if np.sum(state_mask) > 0:
                state_returns.append(np.mean(features[state_mask, 0]))  # mean return for this state
            else:
                state_returns.append(0)
        
        # Score based on which state is more bullish and current probability
        if state_returns[0] > state_returns[1]:
            # State 0 is more bullish
            score = 0.5 + (state_probs[0] * 0.4)  # Max 0.9
        else:
            # State 1 is more bullish
            score = 0.5 + (state_probs[1] * 0.4)  # Max 0.9
            
        # Adjust based on recent trend
        recent_trend = np.mean(returns[-5:])
        if recent_trend > 0:
            score = min(1.0, score + 0.1)
        else:
            score = max(0.0, score - 0.1)
            
        score = float(np.clip(score, 0, 1))
        
        if debug:
            print(f"[DEBUG][HMM] states={len(set(states))}, current_state={recent_state}, "
                  f"state_probs={state_probs}, score={score:.4f}")
        return score
        
    except Exception as e:
        if debug:
            print(f"[DEBUG][HMM] Exception -> {e}, returning neutral score")
        return 0.5

def model_factor_models(df, debug=False):
    """
    Factor Models - מודלים מבוססי גורמים (Fama-French style).
    מתאים במיוחד לטווח ארוך (90 יום) לחיזוי מגמות על בסיס גורמים מרובים.
    """
    try:
        # Simple factor model implementation using multiple regression
        from sklearn.linear_model import LinearRegression
        
        # Create factors
        returns = df['Close'].pct_change().dropna()
        
        # Market factor (overall market movement)
        market_factor = returns.rolling(window=20).mean().dropna()
        
        # Size factor (volume-based)
        volume_factor = (df['Volume'].pct_change().rolling(window=20).mean()).dropna()
        
        # Momentum factor
        momentum_factor = returns.rolling(window=10).sum().dropna()
        
        # Volatility factor
        volatility_factor = returns.rolling(window=20).std().dropna()
        
        # Align all factors to same length
        min_len = min(len(market_factor), len(volume_factor), len(momentum_factor), len(volatility_factor))
        if min_len < 30:
            if debug:
                print("[DEBUG][FactorModels] Insufficient data, returning neutral score")
            return 0.5
        
        # Take last min_len observations
        target_returns = returns[-min_len+1:]  # Next period returns
        factors = np.column_stack([
            market_factor[-min_len:-1],
            volume_factor[-min_len:-1], 
            momentum_factor[-min_len:-1],
            volatility_factor[-min_len:-1]
        ])
        
        # Remove any rows with NaN
        valid_mask = ~np.isnan(factors).any(axis=1) & ~np.isnan(target_returns.values)
        factors = factors[valid_mask]
        target_returns = target_returns.values[valid_mask]
        
        if len(factors) < 20:
            if debug:
                print("[DEBUG][FactorModels] Insufficient clean data, returning neutral score")
            return 0.5
        
        # Fit factor model
        model = LinearRegression()
        model.fit(factors[:-1], target_returns[1:])  # Predict next return from current factors
        
        # Predict next period return
        current_factors = factors[-1:] 
        predicted_return = model.predict(current_factors)[0]
        
        # Convert prediction to score
        # Positive return -> score > 0.5, negative return -> score < 0.5
        score = 0.5 + np.tanh(predicted_return * 10) * 0.4  # Scale and bound to [0.1, 0.9]
        score = float(np.clip(score, 0, 1))
        
        if debug:
            print(f"[DEBUG][FactorModels] predicted_return={predicted_return:.6f}, "
                  f"r2_score={model.score(factors[:-1], target_returns[1:]):.4f}, score={score:.4f}")
        return score
        
    except Exception as e:
        if debug:
            print(f"[DEBUG][FactorModels] Exception -> {e}, returning neutral score")
        return 0.5


# =========================
# Weighting & Classification
# =========================
DEFAULT_WEIGHTS = {
    "XGBoost":     0.15,
    "LSTM":        0.15,
    "GRU":         0.10,
    "Prophet":     0.10,
    "ARIMA_GARCH": 0.10,
    "MonteCarlo":  0.10,
    "Technical":   0.15,
    "Fundamental": 0.15
}

# Time horizon specific weights based on empirical research
# 30 days: ARIMA-GARCH, LSTM, HMM - focus on volatility and short-term patterns
TIME_HORIZON_WEIGHTS_30 = {
    "ARIMA_GARCH": 0.35,  # Most effective for short-term with volatility modeling
    "LSTM":        0.30,  # Good for capturing non-linear short-term patterns
    "HMM":         0.25,  # Effective for detecting sharp market state changes
    "Technical":   0.10,  # Support from technical indicators
    "XGBoost":     0.00,  # Less effective for very short term
    "GRU":         0.00,  # Less effective than LSTM for short term
    "Prophet":     0.00,  # Not optimal for short-term predictions
    "MonteCarlo":  0.00,  # Less reliable for short term
    "Fundamental": 0.00   # Fundamentals less relevant for 30-day predictions
}

# 60 days: LSTM, XGBoost, ARIMA-GARCH - medium-term with long-term dependencies
TIME_HORIZON_WEIGHTS_60 = {
    "LSTM":        0.40,  # Excellent for medium-term dependencies
    "XGBoost":     0.25,  # Strong performance with external factors
    "ARIMA_GARCH": 0.20,  # Still useful but less than LSTM
    "Technical":   0.10,  # Technical analysis support
    "Fundamental": 0.05,  # Some fundamental influence
    "HMM":         0.00,  # Less effective for medium term
    "GRU":         0.00,  # LSTM preferred for this timeframe
    "Prophet":     0.00,  # Not optimal for this specific horizon
    "MonteCarlo":  0.00   # Less reliable for this timeframe
}

# 90 days: Factor Models, LSTM, XGBoost - longer-term trends and multiple factors
TIME_HORIZON_WEIGHTS_90 = {
    "FactorModels": 0.35, # Most effective for longer-term trend prediction
    "LSTM":         0.30, # Strong for long-term prediction as well
    "XGBoost":      0.20, # Good with many external variables
    "Fundamental":  0.10, # Fundamentals become more relevant
    "Technical":    0.05, # Some technical support
    "ARIMA_GARCH":  0.00, # Less effective for longer terms
    "HMM":          0.00, # Not suitable for long-term predictions
    "GRU":          0.00, # LSTM preferred
    "Prophet":      0.00, # Not optimal for this horizon
    "MonteCarlo":   0.00  # Less reliable for longer term
}

def get_time_horizon_weights(time_horizon, debug=False):
    """
    Get the appropriate weights based on time horizon.
    """
    if time_horizon == 30:
        weights = TIME_HORIZON_WEIGHTS_30.copy()
        if debug:
            print(f"[DEBUG] Using 30-day time horizon weights (ARIMA-GARCH, LSTM, HMM focus)")
    elif time_horizon == 60:
        weights = TIME_HORIZON_WEIGHTS_60.copy()
        if debug:
            print(f"[DEBUG] Using 60-day time horizon weights (LSTM, XGBoost, ARIMA-GARCH focus)")
    elif time_horizon == 90:
        weights = TIME_HORIZON_WEIGHTS_90.copy()
        if debug:
            print(f"[DEBUG] Using 90-day time horizon weights (Factor Models, LSTM, XGBoost focus)")
    else:
        weights = DEFAULT_WEIGHTS.copy()
        if debug:
            print(f"[DEBUG] Using default weights (no specific time horizon)")
    
    return weights

def normalize_weights(wdict, debug=False):
    s = sum(wdict.values())
    if s == 0:
        return wdict
    normed = {k: v/s for k,v in wdict.items()}
    if debug:
        print(f"[DEBUG] Normalized weights sum -> {sum(normed.values()):.4f}")
    return normed

def classify(score: float) -> str:
    if score >= 0.70: return "Strong Buy"
    if score >= 0.60: return "Buy"
    if score >= 0.50: return "Hold"
    if score >= 0.40: return "Sell"
    return "Strong Sell"


# =========================
# Core Runner
# =========================
def run_single(ticker: str,
               period: str = "5y",
               interval: str = "1d",
               lookback: int = 80,
               epochs_lstm: int = 6,
               epochs_gru: int = 6,
               fast: bool = False,
               fundamental_override=None,
               custom_weights=None,
               debug=False,
               use_csv_data=None,
               time_horizon=None):

    # Select appropriate weights based on time horizon or custom weights
    if custom_weights:
        weights = custom_weights
    elif time_horizon:
        weights = get_time_horizon_weights(time_horizon, debug=debug)
    else:
        weights = DEFAULT_WEIGHTS
    weights = normalize_weights(weights, debug=debug)

    print(f"\n--- הורדת נתונים: {ticker} period={period} interval={interval} ---")
    
    # Try to use CSV data if ticker is Bitcoin-related or if specified
    if (ticker.upper() in ['BTC-USD', 'BITCOIN', 'BTC'] or use_csv_data) and use_csv_data != False:
        try:
            print("[INFO] Using local CSV data...")
            df = pd.read_csv('BTC-USD.csv', parse_dates=['Date'], index_col='Date')
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"[WARN] Missing columns in CSV: {missing_cols}, using fallback values")
                if 'Open' not in df.columns and 'Close' in df.columns:
                    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
                if 'High' not in df.columns and 'Close' in df.columns:
                    df['High'] = df['Close'] * 1.01
                if 'Low' not in df.columns and 'Close' in df.columns:
                    df['Low'] = df['Close'] * 0.99
                if 'Volume' not in df.columns:
                    df['Volume'] = 1000000  # dummy volume
            df = df.dropna()
            print(f"[INFO] Loaded {len(df)} rows from CSV")
        except Exception as e:
            print(f"[WARN] Failed to load CSV data: {e}, trying yfinance...")
            df = yf.download(ticker, period=period, interval=interval, progress=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if df.empty:
        raise ValueError("לא נמצאו נתונים לטיקר")

    # Feature Engineering
    df = add_technical_indicators(df, debug=debug)
    X, feat_cols = build_feature_matrix(df, debug=debug)

    if fast:
        if debug:
            print("[DEBUG] FAST mode: reducing epochs & sims.")
        epochs_lstm = max(1, min(epochs_lstm, 2))
        epochs_gru = max(1, min(epochs_gru, 2))
        mc_sims = 200
    else:
        mc_sims = 400

    scores = {}

    # Modular safe exec pattern
    def safe_run(name, func):
        try:
            val = func()
        except Exception as e:
            if debug:
                print(f"[DEBUG][{name}] Exception -> {e}")
            val = 0.5
        scores[name] = float(val)

    # מודלי ליבה
    safe_run('XGBoost', lambda: model_xgboost(df, feat_cols, debug=debug))
    safe_run('LSTM', lambda: model_lstm(df, lookback, epochs_lstm, debug=debug))
    safe_run('GRU', lambda: model_gru(df, lookback, epochs_gru, debug=debug))
    safe_run('Prophet', lambda: model_prophet(df, debug=debug))
    safe_run('ARIMA_GARCH', lambda: (model_arima(df, debug=debug) + model_garch(df, debug=debug)) / 2.0)
    safe_run('MonteCarlo', lambda: model_monte_carlo(df, sims=mc_sims, debug=debug))
    safe_run('Technical', lambda: model_technical(df, debug=debug))
    safe_run('Fundamental', lambda: model_fundamental_stub(fundamental_override, debug=debug))
    
    # New time-horizon specific models
    safe_run('HMM', lambda: model_hmm(df, debug=debug))
    safe_run('FactorModels', lambda: model_factor_models(df, debug=debug))

    # Weighted aggregation
    total = 0
    active_wsum = 0
    for k,v in scores.items():
        w = weights.get(k)
        if w is None:
            # מודל בלי משקל מפורש – מדלגים
            continue
        total += v * w
        active_wsum += w

    final_score = total / active_wsum if active_wsum > 0 else 0
    decision = classify(final_score)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if debug:
        print(f"[DEBUG] Active weight sum used: {active_wsum:.4f}")
        print(f"[DEBUG] Final Score Raw: {final_score:.6f}")

    result = {
        "ticker": ticker,
        "timestamp": ts,
        "time_horizon": time_horizon,
        "scores": scores,
        "final_score": final_score,
        "decision": decision,
        "weights_used": {k: weights.get(k, None) for k in scores.keys()}
    }
    return result

def print_report(result: dict):
    print("\n===== תוצאות מודלים =====")
    time_horizon = result.get('time_horizon')
    if time_horizon:
        if time_horizon == 30:
            focus_models = "ARIMA-GARCH, LSTM, HMM"
        elif time_horizon == 60:
            focus_models = "LSTM, XGBoost, ARIMA-GARCH"
        elif time_horizon == 90:
            focus_models = "Factor Models, LSTM, XGBoost"
        print(f"Time Horizon: {time_horizon} days (Focus: {focus_models})")
        print("-------------------------")
    
    for k,v in sorted(result['scores'].items(), key=lambda x: x[0]):
        weight = result['weights_used'].get(k, 0)
        if weight is None:
            weight = 0
        if weight > 0:
            print(f"{k:12s}: {v:.4f} (weight: {weight:.2f})")
        else:
            print(f"{k:12s}: {v:.4f} (not used)")
    print("-------------------------")
    print(f"Final Ensemble Score: {result['final_score']:.4f}")
    print(f"Recommendation: {result['decision']}")
    print(f"Timestamp: {result['timestamp']}")
    print("\n(אין באמור ייעוץ השקעות / Educational Only)")

def save_json(result: dict, path: str):
    try:
        with open(path,'w',encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[INFO] שמור JSON: {path}")
    except Exception as e:
        print(f"[WARN] לא נשמר JSON ({e})")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Single-File Hybrid Ensemble (Detailed)")
    parser.add_argument("--ticker", type=str, default="XLC", help="סימול (טיקר)")
    parser.add_argument("--period", type=str, default="5y", help="טווח (1y/5y/max וכו')")
    parser.add_argument("--interval", type=str, default="1d", help="1d / 1h / 1wk ...")
    parser.add_argument("--lookback", type=int, default=80, help="אורך רצף ל-LSTM/GRU")
    parser.add_argument("--epochs_lstm", type=int, default=6)
    parser.add_argument("--epochs_gru", type=int, default=6)
    parser.add_argument("--fast", action="store_true", help="מצב מהיר (מקטין עומס)")
    parser.add_argument("--fundamental_override", type=float, default=None, help="דריסת ציון פונדמנטלי (0-1)")
    parser.add_argument("--weights", type=str, default=None, help='JSON של משקולות. למשל: {"XGBoost":0.2,...}')
    parser.add_argument("--save_json", type=str, default=None, help="שמור תוצאה כ-JSON")
    parser.add_argument("--debug", action="store_true", help="הדפסות DEBUG מפורטות")
    parser.add_argument("--use_csv", action="store_true", help="Use local CSV data instead of downloading")
    parser.add_argument("--time_horizon", type=int, choices=[30, 60, 90], default=None, help="Time horizon for prediction focus: 30 (short), 60 (medium), 90 (longer-short) days")
    args = parser.parse_args()

    # custom weights parsing
    custom_weights = None
    if args.weights:
        try:
            custom_weights = json.loads(args.weights)
        except Exception:
            print("אזהרה: פורמט weights לא תקין – מתעלם.")

    result = run_single(
        ticker=args.ticker.upper(),
        period=args.period,
        interval=args.interval,
        lookback=args.lookback,
        epochs_lstm=args.epochs_lstm,
        epochs_gru=args.epochs_gru,
        fast=args.fast,
        fundamental_override=args.fundamental_override,
        custom_weights=custom_weights,
        debug=args.debug,
        use_csv_data=args.use_csv,
        time_horizon=args.time_horizon
    )
    print_report(result)

    if args.save_json:
        save_json(result, args.save_json)

if __name__ == "__main__":
    main()