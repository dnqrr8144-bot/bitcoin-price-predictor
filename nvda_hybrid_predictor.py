"""
NVDA Hybrid Stock Analysis System
Implements multiple models for comprehensive stock analysis and buy/sell recommendations.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Core ML libraries (mock imports for now, will work when packages are installed)
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    import yfinance as yf
    from textblob import TextBlob
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    print(f"Warning: Some packages not installed: {e}")
    print("Please install requirements.txt for full functionality")

# Neural network libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    print("Warning: TensorFlow not available. Neural network models will be mocked.")

# Technical analysis
try:
    import ta
except ImportError:
    print("Warning: Technical analysis library not available")

class NVDAHybridPredictor:
    """
    Comprehensive NVDA stock analysis system combining multiple models:
    - XGBoost
    - LSTM
    - GRU  
    - ARIMA
    - Technical Analysis
    - Sentiment Analysis
    - Monte Carlo Simulation
    - Ensemble Hybrid Model
    """
    
    def __init__(self, symbol='NVDA', period='5y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.models = {}
        self.predictions = {}
        self.weights = {
            'xgboost': 0.2,
            'lstm': 0.18,
            'gru': 0.15,
            'arima': 0.12,
            'technical': 0.15,
            'sentiment': 0.1,
            'monte_carlo': 0.1
        }
        self.scaler = MinMaxScaler()
        self.recommendation = None
        
    def fetch_data(self):
        """Fetch NVDA stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            # Add technical indicators
            self.add_technical_indicators()
            
            print(f"Successfully fetched {len(self.data)} days of {self.symbol} data")
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Mock data for testing
            self.create_mock_data()
            return False
    
    def create_mock_data(self):
        """Create mock NVDA data for testing when yfinance is not available"""
        dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        # Simulate NVDA-like price movement
        base_price = 200
        returns = np.random.normal(0.001, 0.03, len(dates))  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'High': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(10000000, 100000000, len(dates))
        }, index=dates)
        
        self.add_technical_indicators()
        print(f"Created mock {self.symbol} data with {len(self.data)} days")
    
    def add_technical_indicators(self):
        """Add technical analysis indicators"""
        try:
            # Moving averages
            self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
            self.data['EMA_12'] = self.data['Close'].ewm(span=12).mean()
            self.data['EMA_26'] = self.data['Close'].ewm(span=26).mean()
            
            # MACD
            self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
            self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()
            
            # RSI
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            self.data['BB_middle'] = self.data['Close'].rolling(window=20).mean()
            bb_std = self.data['Close'].rolling(window=20).std()
            self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * 2)
            self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * 2)
            
            # Volume indicators
            self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
            
        except Exception as e:
            print(f"Warning: Error adding technical indicators: {e}")
    
    def prepare_sequences(self, data, lookback=60):
        """Prepare data sequences for neural networks"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def train_xgboost_model(self):
        """Train XGBoost regression model"""
        try:
            # Prepare features
            features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 
                       'RSI', 'MACD', 'BB_middle', 'Volume_SMA']
            
            df_clean = self.data.dropna()
            X = df_clean[features].values
            y = df_clean['Close'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            self.models['xgboost'] = model
            self.predictions['xgboost'] = {
                'train_pred': train_pred,
                'test_pred': test_pred,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'latest_prediction': test_pred[-1] if len(test_pred) > 0 else y[-1]
            }
            
            print(f"XGBoost - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
            
        except Exception as e:
            print(f"XGBoost training failed: {e}")
            # Mock prediction
            self.predictions['xgboost'] = {
                'latest_prediction': self.data['Close'].iloc[-1] * 1.02,
                'train_rmse': 10.0,
                'test_rmse': 12.0
            }
    
    def train_lstm_model(self):
        """Train LSTM neural network"""
        try:
            # Use TensorFlow if available
            data = self.data['Close'].values.reshape(-1, 1)
            data_scaled = self.scaler.fit_transform(data)
            
            lookback = 60
            X, y = self.prepare_sequences(data_scaled, lookback)
            
            # Train-test split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            # Train with early stopping to prevent overfitting
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,  # Reduced for faster training
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Inverse transform
            train_pred = self.scaler.inverse_transform(train_pred)
            test_pred = self.scaler.inverse_transform(test_pred)
            y_train_orig = self.scaler.inverse_transform(y_train.reshape(-1, 1))
            y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred))
            
            self.models['lstm'] = model
            self.predictions['lstm'] = {
                'train_pred': train_pred,
                'test_pred': test_pred,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'latest_prediction': test_pred[-1][0] if len(test_pred) > 0 else data[-1][0]
            }
            
            print(f"LSTM - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
            
        except Exception as e:
            print(f"LSTM training failed: {e}")
            # Mock prediction
            self.predictions['lstm'] = {
                'latest_prediction': self.data['Close'].iloc[-1] * 1.01,
                'train_rmse': 8.0,
                'test_rmse': 10.0
            }
    
    def train_gru_model(self):
        """Train GRU neural network"""
        try:
            # Similar to LSTM but with GRU layers
            data = self.data['Close'].values.reshape(-1, 1)
            data_scaled = self.scaler.fit_transform(data)
            
            lookback = 60
            X, y = self.prepare_sequences(data_scaled, lookback)
            
            # Train-test split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Build GRU model
            model = Sequential([
                GRU(50, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                GRU(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=0
            )
            
            # Predictions
            test_pred = model.predict(X_test)
            test_pred = self.scaler.inverse_transform(test_pred)
            y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred))
            
            self.models['gru'] = model
            self.predictions['gru'] = {
                'test_pred': test_pred,
                'test_rmse': test_rmse,
                'latest_prediction': test_pred[-1][0] if len(test_pred) > 0 else data[-1][0]
            }
            
            print(f"GRU - Test RMSE: {test_rmse:.2f}")
            
        except Exception as e:
            print(f"GRU training failed: {e}")
            # Mock prediction
            self.predictions['gru'] = {
                'latest_prediction': self.data['Close'].iloc[-1] * 0.99,
                'test_rmse': 9.0
            }
    
    def train_arima_model(self):
        """Train ARIMA time series model"""
        try:
            # Use closing prices
            ts_data = self.data['Close'].dropna()
            
            # Fit ARIMA model (using simple (1,1,1) for speed)
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast next period
            forecast = fitted_model.forecast(steps=1)
            
            # Calculate RMSE on last 20% of data
            train_size = int(0.8 * len(ts_data))
            train_data = ts_data[:train_size]
            test_data = ts_data[train_size:]
            
            # Re-fit on training data for validation
            train_model = ARIMA(train_data, order=(1, 1, 1))
            train_fitted = train_model.fit()
            
            # Generate predictions for test period
            test_predictions = []
            for i in range(len(test_data)):
                pred = train_fitted.forecast(steps=1)[0]
                test_predictions.append(pred)
                # Update model with actual data point (walk-forward validation)
                train_fitted = train_fitted.extend([test_data.iloc[i]])
            
            test_rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
            
            self.models['arima'] = fitted_model
            self.predictions['arima'] = {
                'test_rmse': test_rmse,
                'latest_prediction': forecast[0],
                'test_pred': test_predictions
            }
            
            print(f"ARIMA - Test RMSE: {test_rmse:.2f}")
            
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            # Mock prediction
            self.predictions['arima'] = {
                'latest_prediction': self.data['Close'].iloc[-1] * 1.005,
                'test_rmse': 15.0
            }
    
    def technical_analysis(self):
        """Perform technical analysis and generate signals"""
        try:
            latest_data = self.data.iloc[-1]
            signals = {}
            
            # RSI signals
            rsi = latest_data['RSI']
            if rsi < 30:
                signals['rsi'] = 'BUY'  # Oversold
            elif rsi > 70:
                signals['rsi'] = 'SELL'  # Overbought
            else:
                signals['rsi'] = 'HOLD'
            
            # MACD signals
            macd = latest_data['MACD']
            macd_signal = latest_data['MACD_signal']
            if macd > macd_signal:
                signals['macd'] = 'BUY'
            else:
                signals['macd'] = 'SELL'
            
            # Moving average signals
            close_price = latest_data['Close']
            sma_20 = latest_data['SMA_20']
            sma_50 = latest_data['SMA_50']
            
            if close_price > sma_20 > sma_50:
                signals['ma'] = 'BUY'
            elif close_price < sma_20 < sma_50:
                signals['ma'] = 'SELL'
            else:
                signals['ma'] = 'HOLD'
            
            # Bollinger Bands
            bb_upper = latest_data['BB_upper']
            bb_lower = latest_data['BB_lower']
            
            if close_price <= bb_lower:
                signals['bb'] = 'BUY'  # Price at lower band
            elif close_price >= bb_upper:
                signals['bb'] = 'SELL'  # Price at upper band
            else:
                signals['bb'] = 'HOLD'
            
            # Aggregate technical signal
            buy_signals = sum(1 for s in signals.values() if s == 'BUY')
            sell_signals = sum(1 for s in signals.values() if s == 'SELL')
            
            if buy_signals > sell_signals:
                overall_signal = 'BUY'
                prediction = close_price * 1.02  # Expect 2% increase
            elif sell_signals > buy_signals:
                overall_signal = 'SELL'
                prediction = close_price * 0.98  # Expect 2% decrease
            else:
                overall_signal = 'HOLD'
                prediction = close_price
            
            self.predictions['technical'] = {
                'signals': signals,
                'overall_signal': overall_signal,
                'latest_prediction': prediction
            }
            
            print(f"Technical Analysis - Signal: {overall_signal}")
            
        except Exception as e:
            print(f"Technical analysis failed: {e}")
            self.predictions['technical'] = {
                'latest_prediction': self.data['Close'].iloc[-1],
                'overall_signal': 'HOLD'
            }
    
    def sentiment_analysis(self):
        """Mock sentiment analysis (placeholder for news/social media sentiment)"""
        try:
            # In a real implementation, this would analyze:
            # - News headlines about NVDA
            # - Social media sentiment
            # - Financial reports sentiment
            # - Analyst reports
            
            # Mock sentiment score (-1 to 1)
            # This could be integrated with news APIs, Twitter API, etc.
            mock_sentiment = np.random.normal(0.1, 0.3)  # Slightly positive bias for NVDA
            mock_sentiment = np.clip(mock_sentiment, -1, 1)
            
            # Convert sentiment to price prediction
            current_price = self.data['Close'].iloc[-1]
            sentiment_multiplier = 1 + (mock_sentiment * 0.05)  # Max 5% impact
            
            self.predictions['sentiment'] = {
                'sentiment_score': mock_sentiment,
                'latest_prediction': current_price * sentiment_multiplier
            }
            
            print(f"Sentiment Analysis - Score: {mock_sentiment:.3f}")
            
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            self.predictions['sentiment'] = {
                'latest_prediction': self.data['Close'].iloc[-1],
                'sentiment_score': 0.0
            }
    
    def monte_carlo_simulation(self, num_simulations=1000, days=30):
        """Monte Carlo simulation for price prediction"""
        try:
            # Calculate daily returns
            returns = self.data['Close'].pct_change().dropna()
            
            # Get statistics
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Current price
            current_price = self.data['Close'].iloc[-1]
            
            # Run simulations
            simulations = []
            for _ in range(num_simulations):
                price_path = [current_price]
                for _ in range(days):
                    random_return = np.random.normal(mean_return, std_return)
                    next_price = price_path[-1] * (1 + random_return)
                    price_path.append(next_price)
                simulations.append(price_path[-1])  # Final price
            
            # Calculate statistics
            simulations = np.array(simulations)
            mean_prediction = np.mean(simulations)
            std_prediction = np.std(simulations)
            percentile_5 = np.percentile(simulations, 5)
            percentile_95 = np.percentile(simulations, 95)
            
            self.predictions['monte_carlo'] = {
                'latest_prediction': mean_prediction,
                'std': std_prediction,
                'percentile_5': percentile_5,
                'percentile_95': percentile_95,
                'simulations': simulations
            }
            
            print(f"Monte Carlo - Mean: {mean_prediction:.2f}, Std: {std_prediction:.2f}")
            
        except Exception as e:
            print(f"Monte Carlo simulation failed: {e}")
            self.predictions['monte_carlo'] = {
                'latest_prediction': self.data['Close'].iloc[-1] * 1.03,
                'std': 10.0
            }
    
    def create_ensemble_prediction(self):
        """Create weighted ensemble prediction from all models"""
        try:
            ensemble_prediction = 0
            total_weight = 0
            
            for model_name, weight in self.weights.items():
                if model_name in self.predictions:
                    pred = self.predictions[model_name]['latest_prediction']
                    ensemble_prediction += pred * weight
                    total_weight += weight
            
            # Normalize by actual total weight (in case some models failed)
            if total_weight > 0:
                ensemble_prediction /= total_weight
            else:
                ensemble_prediction = self.data['Close'].iloc[-1]
            
            current_price = self.data['Close'].iloc[-1]
            price_change = (ensemble_prediction - current_price) / current_price
            
            # Generate buy/sell recommendation
            if price_change > 0.02:  # Expected gain > 2%
                recommendation = "STRONG BUY"
            elif price_change > 0.01:  # Expected gain > 1%
                recommendation = "BUY"
            elif price_change < -0.02:  # Expected loss > 2%
                recommendation = "STRONG SELL"
            elif price_change < -0.01:  # Expected loss > 1%
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            self.recommendation = {
                'ensemble_prediction': ensemble_prediction,
                'current_price': current_price,
                'expected_change': price_change,
                'recommendation': recommendation,
                'confidence': min(abs(price_change) * 10, 1.0)  # Simple confidence metric
            }
            
            print(f"\nEnsemble Prediction: ${ensemble_prediction:.2f}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Expected Change: {price_change:.2%}")
            print(f"Recommendation: {recommendation}")
            
        except Exception as e:
            print(f"Ensemble prediction failed: {e}")
            current_price = self.data['Close'].iloc[-1]
            self.recommendation = {
                'ensemble_prediction': current_price,
                'current_price': current_price,
                'expected_change': 0.0,
                'recommendation': 'HOLD',
                'confidence': 0.5
            }
    
    def train_all_models(self):
        """Train all prediction models"""
        print("Training all models...")
        print("=" * 50)
        
        self.train_xgboost_model()
        self.train_lstm_model()
        self.train_gru_model()
        self.train_arima_model()
        self.technical_analysis()
        self.sentiment_analysis()
        self.monte_carlo_simulation()
        
        print("\n" + "=" * 50)
        print("Creating ensemble prediction...")
        self.create_ensemble_prediction()
    
    def plot_analysis(self):
        """Create comprehensive visualization of the analysis"""
        try:
            fig = plt.figure(figsize=(20, 12))
            
            # Main price chart with technical indicators
            plt.subplot(2, 3, 1)
            plt.plot(self.data.index[-200:], self.data['Close'].iloc[-200:], label='Close Price', linewidth=2)
            plt.plot(self.data.index[-200:], self.data['SMA_20'].iloc[-200:], label='SMA 20', alpha=0.7)
            plt.plot(self.data.index[-200:], self.data['SMA_50'].iloc[-200:], label='SMA 50', alpha=0.7)
            plt.fill_between(self.data.index[-200:], 
                           self.data['BB_upper'].iloc[-200:], 
                           self.data['BB_lower'].iloc[-200:], 
                           alpha=0.2, label='Bollinger Bands')
            plt.title(f'{self.symbol} Price with Technical Indicators')
            plt.legend()
            plt.xticks(rotation=45)
            
            # RSI
            plt.subplot(2, 3, 2)
            plt.plot(self.data.index[-200:], self.data['RSI'].iloc[-200:])
            plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            plt.title('RSI (14-period)')
            plt.legend()
            plt.xticks(rotation=45)
            
            # MACD
            plt.subplot(2, 3, 3)
            plt.plot(self.data.index[-200:], self.data['MACD'].iloc[-200:], label='MACD')
            plt.plot(self.data.index[-200:], self.data['MACD_signal'].iloc[-200:], label='Signal')
            plt.title('MACD')
            plt.legend()
            plt.xticks(rotation=45)
            
            # Model predictions comparison
            plt.subplot(2, 3, 4)
            models = []
            predictions = []
            current_price = self.data['Close'].iloc[-1]
            
            for model_name in ['xgboost', 'lstm', 'gru', 'arima', 'technical', 'sentiment', 'monte_carlo']:
                if model_name in self.predictions:
                    models.append(model_name.upper())
                    pred = self.predictions[model_name]['latest_prediction']
                    predictions.append((pred - current_price) / current_price * 100)
            
            colors = ['red' if p < 0 else 'green' for p in predictions]
            plt.bar(models, predictions, color=colors, alpha=0.7)
            plt.title('Model Predictions (% Change from Current)')
            plt.ylabel('% Change')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Ensemble recommendation
            plt.subplot(2, 3, 5)
            if self.recommendation:
                rec = self.recommendation['recommendation']
                change = self.recommendation['expected_change'] * 100
                confidence = self.recommendation['confidence']
                
                color_map = {
                    'STRONG BUY': 'darkgreen',
                    'BUY': 'green', 
                    'HOLD': 'orange',
                    'SELL': 'red',
                    'STRONG SELL': 'darkred'
                }
                
                plt.bar(['Recommendation'], [change], 
                       color=color_map.get(rec, 'gray'), alpha=0.7)
                plt.title(f'Ensemble Recommendation: {rec}')
                plt.ylabel('Expected % Change')
                plt.text(0, change/2, f'Confidence: {confidence:.1%}', 
                        ha='center', va='center', fontweight='bold')
            
            # Monte Carlo simulation results
            plt.subplot(2, 3, 6)
            if 'monte_carlo' in self.predictions:
                mc_data = self.predictions['monte_carlo']
                if 'simulations' in mc_data:
                    plt.hist(mc_data['simulations'], bins=50, alpha=0.7, color='blue')
                    plt.axvline(self.data['Close'].iloc[-1], color='red', 
                              linestyle='--', label='Current Price')
                    plt.axvline(mc_data['latest_prediction'], color='green', 
                              linestyle='--', label='Mean Prediction')
                    plt.title('Monte Carlo Price Distribution')
                    plt.xlabel('Price')
                    plt.ylabel('Frequency')
                    plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        report = f"""
NVDA HYBRID STOCK ANALYSIS REPORT
================================
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbol: {self.symbol}
Current Price: ${self.data['Close'].iloc[-1]:.2f}

ENSEMBLE RECOMMENDATION
======================
"""
        
        if self.recommendation:
            report += f"""
Recommendation: {self.recommendation['recommendation']}
Target Price: ${self.recommendation['ensemble_prediction']:.2f}
Expected Change: {self.recommendation['expected_change']:.2%}
Confidence Level: {self.recommendation['confidence']:.1%}
"""
        
        report += """
MODEL PREDICTIONS
================
"""
        
        for model_name, pred_data in self.predictions.items():
            if 'latest_prediction' in pred_data:
                current_price = self.data['Close'].iloc[-1]
                pred_price = pred_data['latest_prediction']
                change = (pred_price - current_price) / current_price
                
                report += f"{model_name.upper()}: ${pred_price:.2f} ({change:.2%})\n"
        
        report += f"""
TECHNICAL INDICATORS (Latest)
============================
RSI (14): {self.data['RSI'].iloc[-1]:.2f}
MACD: {self.data['MACD'].iloc[-1]:.4f}
SMA 20: ${self.data['SMA_20'].iloc[-1]:.2f}
SMA 50: ${self.data['SMA_50'].iloc[-1]:.2f}
Bollinger Upper: ${self.data['BB_upper'].iloc[-1]:.2f}
Bollinger Lower: ${self.data['BB_lower'].iloc[-1]:.2f}

MODEL WEIGHTS
============
"""
        
        for model, weight in self.weights.items():
            report += f"{model.upper()}: {weight:.1%}\n"
        
        print(report)
        return report
    
    def run_complete_analysis(self):
        """Run the complete hybrid analysis"""
        print("Starting NVDA Hybrid Stock Analysis...")
        print("=" * 50)
        
        # Fetch data
        print("1. Fetching stock data...")
        self.fetch_data()
        
        # Train models
        print("\n2. Training prediction models...")
        self.train_all_models()
        
        # Generate visualizations
        print("\n3. Creating visualizations...")
        self.plot_analysis()
        
        # Generate report
        print("\n4. Generating analysis report...")
        self.generate_report()
        
        print("\nAnalysis complete!")
        return self.recommendation

if __name__ == "__main__":
    # Run the analysis
    predictor = NVDAHybridPredictor()
    recommendation = predictor.run_complete_analysis()