#!/usr/bin/env python3
"""
Lightweight NVDA Analysis System Test
Works without external dependencies for demonstration
"""

import random
import math
from datetime import datetime, timedelta

class NVDALitePredictor:
    """
    Lightweight version of NVDA predictor for testing without dependencies
    """
    
    def __init__(self, symbol='NVDA'):
        self.symbol = symbol
        self.data = []
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
        self.recommendation = None
        
    def create_mock_data(self, days=500):
        """Create mock NVDA price data"""
        random.seed(42)
        
        # Start with realistic NVDA price
        base_price = 450
        prices = []
        dates = []
        
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            # Simulate daily price movement
            daily_change = random.gauss(0.002, 0.03)  # Mean 0.2% daily gain, 3% volatility
            if i == 0:
                price = base_price
            else:
                price = prices[-1] * (1 + daily_change)
            
            prices.append(price)
            dates.append(start_date + timedelta(days=i))
        
        # Create data structure
        for i, (date, price) in enumerate(zip(dates, prices)):
            volume = random.randint(20000000, 150000000)
            high = price * random.uniform(1.0, 1.03)
            low = price * random.uniform(0.97, 1.0)
            open_price = price * random.uniform(0.99, 1.01)
            
            self.data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        print(f"Created mock {self.symbol} data: {len(self.data)} days")
        print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"Current price: ${prices[-1]:.2f}")
    
    def calculate_sma(self, period=20):
        """Calculate Simple Moving Average"""
        if len(self.data) < period:
            return None
        
        prices = [d['close'] for d in self.data[-period:]]
        return sum(prices) / len(prices)
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        if len(self.data) < period + 1:
            return 50  # Neutral
        
        gains = []
        losses = []
        
        for i in range(1, period + 1):
            change = self.data[-i]['close'] - self.data[-i-1]['close']
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def mock_xgboost_prediction(self):
        """Mock XGBoost model prediction"""
        current_price = self.data[-1]['close']
        
        # Simulate XGBoost considering recent trend
        recent_prices = [d['close'] for d in self.data[-10:]]
        trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Add some randomness
        prediction = current_price * (1 + trend * 0.5 + random.gauss(0, 0.02))
        
        self.predictions['xgboost'] = {
            'latest_prediction': prediction,
            'confidence': 0.75
        }
        
        print(f"XGBoost prediction: ${prediction:.2f}")
    
    def mock_lstm_prediction(self):
        """Mock LSTM neural network prediction"""
        current_price = self.data[-1]['close']
        
        # LSTM typically captures longer-term patterns
        long_term_prices = [d['close'] for d in self.data[-60:]]
        long_trend = (long_term_prices[-1] - long_term_prices[0]) / long_term_prices[0]
        
        prediction = current_price * (1 + long_trend * 0.3 + random.gauss(0, 0.015))
        
        self.predictions['lstm'] = {
            'latest_prediction': prediction,
            'confidence': 0.80
        }
        
        print(f"LSTM prediction: ${prediction:.2f}")
    
    def mock_gru_prediction(self):
        """Mock GRU neural network prediction"""
        current_price = self.data[-1]['close']
        
        # GRU similar to LSTM but slightly different
        prediction = current_price * (1 + random.gauss(0.01, 0.018))
        
        self.predictions['gru'] = {
            'latest_prediction': prediction,
            'confidence': 0.78
        }
        
        print(f"GRU prediction: ${prediction:.2f}")
    
    def mock_arima_prediction(self):
        """Mock ARIMA time series prediction"""
        current_price = self.data[-1]['close']
        
        # ARIMA focuses on time series patterns
        # Simple trend continuation
        recent_changes = []
        for i in range(1, 8):  # Last week
            change = (self.data[-i]['close'] - self.data[-i-1]['close']) / self.data[-i-1]['close']
            recent_changes.append(change)
        
        avg_change = sum(recent_changes) / len(recent_changes)
        prediction = current_price * (1 + avg_change)
        
        self.predictions['arima'] = {
            'latest_prediction': prediction,
            'confidence': 0.65
        }
        
        print(f"ARIMA prediction: ${prediction:.2f}")
    
    def technical_analysis(self):
        """Perform technical analysis"""
        current_price = self.data[-1]['close']
        sma_20 = self.calculate_sma(20)
        sma_50 = self.calculate_sma(50)
        rsi = self.calculate_rsi(14)
        
        signals = []
        
        # RSI signals
        if rsi < 30:
            signals.append('buy')  # Oversold
        elif rsi > 70:
            signals.append('sell')  # Overbought
        else:
            signals.append('hold')
        
        # Moving average signals
        if sma_20 and sma_50:
            if current_price > sma_20 > sma_50:
                signals.append('buy')
            elif current_price < sma_20 < sma_50:
                signals.append('sell')
            else:
                signals.append('hold')
        
        # Volume analysis (simplified)
        recent_volumes = [d['volume'] for d in self.data[-5:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = self.data[-1]['volume']
        
        volume_signal = 'high' if current_volume > avg_volume * 1.2 else 'normal'
        
        # Aggregate signals
        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')
        
        if buy_signals > sell_signals:
            overall_signal = 'BUY'
            prediction = current_price * 1.02
        elif sell_signals > buy_signals:
            overall_signal = 'SELL'  
            prediction = current_price * 0.98
        else:
            overall_signal = 'HOLD'
            prediction = current_price
        
        self.predictions['technical'] = {
            'latest_prediction': prediction,
            'signal': overall_signal,
            'rsi': rsi,
            'volume_signal': volume_signal
        }
        
        print(f"Technical Analysis - Signal: {overall_signal}, RSI: {rsi:.1f}")
    
    def sentiment_analysis(self):
        """Mock sentiment analysis"""
        current_price = self.data[-1]['close']
        
        # Mock sentiment score (in reality would come from news/social media)
        sentiment_score = random.gauss(0.15, 0.25)  # Slightly positive bias for NVDA
        sentiment_score = max(-1, min(1, sentiment_score))  # Clamp to [-1, 1]
        
        # Convert sentiment to price impact
        sentiment_impact = sentiment_score * 0.03  # Max 3% impact
        prediction = current_price * (1 + sentiment_impact)
        
        self.predictions['sentiment'] = {
            'latest_prediction': prediction,
            'sentiment_score': sentiment_score
        }
        
        print(f"Sentiment Analysis - Score: {sentiment_score:.3f}")
    
    def monte_carlo_simulation(self, simulations=1000):
        """Mock Monte Carlo simulation"""
        current_price = self.data[-1]['close']
        
        # Calculate historical volatility
        returns = []
        for i in range(1, min(100, len(self.data))):
            ret = (self.data[-i]['close'] - self.data[-i-1]['close']) / self.data[-i-1]['close']
            returns.append(ret)
        
        mean_return = sum(returns) / len(returns)
        
        # Calculate standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance)
        
        # Run simulations
        simulation_results = []
        for _ in range(simulations):
            # 30-day forecast
            price = current_price
            for _ in range(30):
                daily_return = random.gauss(mean_return, std_return)
                price *= (1 + daily_return)
            simulation_results.append(price)
        
        # Calculate statistics
        mean_prediction = sum(simulation_results) / len(simulation_results)
        
        self.predictions['monte_carlo'] = {
            'latest_prediction': mean_prediction,
            'simulations': simulation_results
        }
        
        print(f"Monte Carlo - Mean prediction: ${mean_prediction:.2f}")
    
    def create_ensemble_prediction(self):
        """Create weighted ensemble prediction"""
        ensemble_prediction = 0
        total_weight = 0
        
        # Combine all model predictions with weights
        for model_name, weight in self.weights.items():
            if model_name in self.predictions:
                pred = self.predictions[model_name]['latest_prediction']
                ensemble_prediction += pred * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_prediction /= total_weight
        else:
            ensemble_prediction = self.data[-1]['close']
        
        current_price = self.data[-1]['close']
        price_change = (ensemble_prediction - current_price) / current_price
        
        # Generate recommendation
        if price_change > 0.03:
            recommendation = "STRONG BUY"
        elif price_change > 0.015:
            recommendation = "BUY"
        elif price_change < -0.03:
            recommendation = "STRONG SELL"
        elif price_change < -0.015:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        confidence = min(abs(price_change) * 20, 1.0)
        
        self.recommendation = {
            'ensemble_prediction': ensemble_prediction,
            'current_price': current_price,
            'expected_change': price_change,
            'recommendation': recommendation,
            'confidence': confidence
        }
        
        print(f"\nEnsemble Results:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${ensemble_prediction:.2f}")
        print(f"Expected Change: {price_change:.2%}")
        print(f"Recommendation: {recommendation}")
        print(f"Confidence: {confidence:.1%}")
    
    def generate_report(self):
        """Generate analysis report"""
        current_price = self.data[-1]['close']
        rsi = self.calculate_rsi()
        sma_20 = self.calculate_sma(20)
        
        report = f"""
NVDA HYBRID STOCK ANALYSIS REPORT
=================================
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Symbol: {self.symbol}
Current Price: ${current_price:.2f}

ENSEMBLE RECOMMENDATION
======================
Recommendation: {self.recommendation['recommendation']}
Target Price: ${self.recommendation['ensemble_prediction']:.2f}
Expected Change: {self.recommendation['expected_change']:.2%}
Confidence Level: {self.recommendation['confidence']:.1%}

MODEL PREDICTIONS
================
"""
        
        for model_name, pred_data in self.predictions.items():
            pred_price = pred_data['latest_prediction']
            change = (pred_price - current_price) / current_price
            report += f"{model_name.upper()}: ${pred_price:.2f} ({change:+.2%})\n"
        
        report += f"""
TECHNICAL INDICATORS
===================
RSI (14): {rsi:.1f}
SMA (20): ${sma_20:.2f}
Volume Signal: {self.predictions.get('technical', {}).get('volume_signal', 'N/A')}

MODEL WEIGHTS
============
"""
        
        for model, weight in self.weights.items():
            report += f"{model.upper()}: {weight:.1%}\n"
        
        print(report)
        return report
    
    def run_analysis(self):
        """Run complete analysis"""
        print("NVDA Hybrid Stock Analysis System")
        print("=" * 50)
        
        # Create mock data
        print("1. Creating market data...")
        self.create_mock_data()
        
        # Run all models
        print("\n2. Running prediction models...")
        self.mock_xgboost_prediction()
        self.mock_lstm_prediction()
        self.mock_gru_prediction()
        self.mock_arima_prediction()
        self.technical_analysis()
        self.sentiment_analysis()
        self.monte_carlo_simulation()
        
        # Create ensemble
        print("\n3. Creating ensemble prediction...")
        self.create_ensemble_prediction()
        
        # Generate report
        print("\n4. Generating report...")
        self.generate_report()
        
        print("\nAnalysis complete!")
        return self.recommendation

def test_system():
    """Test the NVDA analysis system"""
    try:
        predictor = NVDALitePredictor()
        result = predictor.run_analysis()
        
        print("\n" + "=" * 50)
        print("✓ System test successful!")
        print("All models executed and ensemble recommendation generated.")
        
        return True
        
    except Exception as e:
        print(f"✗ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_system()