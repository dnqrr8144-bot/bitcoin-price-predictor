#!/usr/bin/env python3
"""
Test script for NVDA Hybrid Prediction System
Tests the system with mock data when packages are not available
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_nvda_system():
    """Test the NVDA hybrid prediction system"""
    try:
        from nvda_hybrid_predictor import NVDAHybridPredictor
        
        print("Testing NVDA Hybrid Prediction System")
        print("=" * 50)
        
        # Initialize predictor
        predictor = NVDAHybridPredictor(symbol='NVDA', period='2y')
        
        # Test data fetching (will use mock data if yfinance not available)
        print("Testing data fetching...")
        predictor.fetch_data()
        
        if predictor.data is not None:
            print(f"✓ Data fetched successfully: {len(predictor.data)} rows")
            print(f"✓ Date range: {predictor.data.index[0]} to {predictor.data.index[-1]}")
            print(f"✓ Current price: ${predictor.data['Close'].iloc[-1]:.2f}")
        else:
            print("✗ Data fetching failed")
            return False
        
        # Test individual models
        print("\nTesting individual models...")
        
        print("- Testing XGBoost...")
        predictor.train_xgboost_model()
        
        print("- Testing LSTM...")
        predictor.train_lstm_model()
        
        print("- Testing GRU...")
        predictor.train_gru_model()
        
        print("- Testing ARIMA...")
        predictor.train_arima_model()
        
        print("- Testing Technical Analysis...")
        predictor.technical_analysis()
        
        print("- Testing Sentiment Analysis...")
        predictor.sentiment_analysis()
        
        print("- Testing Monte Carlo...")
        predictor.monte_carlo_simulation()
        
        # Test ensemble
        print("\nTesting ensemble prediction...")
        predictor.create_ensemble_prediction()
        
        if predictor.recommendation:
            print("✓ Ensemble prediction created successfully")
            print(f"  Recommendation: {predictor.recommendation['recommendation']}")
            print(f"  Target Price: ${predictor.recommendation['ensemble_prediction']:.2f}")
            print(f"  Expected Change: {predictor.recommendation['expected_change']:.2%}")
        else:
            print("✗ Ensemble prediction failed")
            return False
        
        # Test report generation
        print("\nTesting report generation...")
        report = predictor.generate_report()
        
        if report:
            print("✓ Report generated successfully")
        else:
            print("✗ Report generation failed")
            return False
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("The NVDA Hybrid Prediction System is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nvda_system()
    sys.exit(0 if success else 1)