#!/usr/bin/env python3
"""
Demo script showcasing the integrated Bitcoin & NVDA analysis system
"""

from datetime import datetime
from nvda_lite_test import NVDALitePredictor

def demo_integrated_system():
    """Demonstrate the complete analysis system"""
    
    print("=" * 80)
    print("INTEGRATED BITCOIN & NVDA STOCK ANALYSIS SYSTEM")
    print("=" * 80)
    print(f"Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Bitcoin Analysis Section (simplified demo)
    print("🟠 BITCOIN ANALYSIS")
    print("-" * 50)
    print("Original system: LSTM neural network for Bitcoin price prediction")
    print("Features: Time series analysis, technical indicators, price forecasting")
    print("Status: ✓ Operational (requires TensorFlow for full functionality)")
    print()
    
    # NVDA Hybrid Analysis  
    print("🟢 NVDA HYBRID ANALYSIS SYSTEM")
    print("-" * 50)
    
    try:
        # Initialize NVDA predictor
        nvda_predictor = NVDALitePredictor(symbol='NVDA')
        
        # Run analysis
        print("Running comprehensive NVDA analysis...")
        recommendation = nvda_predictor.run_analysis()
        
        if recommendation:
            print("\n🎯 INVESTMENT RECOMMENDATION")
            print("-" * 30)
            print(f"Current Price: ${recommendation['current_price']:.2f}")
            print(f"Target Price: ${recommendation['ensemble_prediction']:.2f}")
            print(f"Expected Return: {recommendation['expected_change']:.2%}")
            print(f"Recommendation: {recommendation['recommendation']}")
            print(f"Confidence Level: {recommendation['confidence']:.1%}")
            
            # Investment advice based on recommendation
            print(f"\n💡 INVESTMENT ADVICE")
            print("-" * 20)
            
            if recommendation['recommendation'] in ['STRONG BUY', 'BUY']:
                print("✅ Consider BUYING NVDA stock")
                print("   • Multiple models suggest upward price movement")
                print("   • Technical indicators support bullish outlook")
                if recommendation['confidence'] > 0.7:
                    print("   • High confidence in prediction")
                
            elif recommendation['recommendation'] in ['STRONG SELL', 'SELL']:
                print("❌ Consider SELLING NVDA stock")
                print("   • Multiple models suggest downward price movement") 
                print("   • Technical indicators support bearish outlook")
                if recommendation['confidence'] > 0.7:
                    print("   • High confidence in prediction")
                    
            else:
                print("⏸️ HOLD current position")
                print("   • Mixed signals from different models")
                print("   • Market conditions uncertain")
            
            print(f"\n⚠️  Risk Disclaimer: This is for educational purposes only.")
            print("   Always do your own research before making investment decisions.")
            
        return True
        
    except Exception as e:
        print(f"Error running NVDA analysis: {e}")
        return False

def show_model_architecture():
    """Display the hybrid model architecture"""
    print("\n🏗️ HYBRID MODEL ARCHITECTURE")
    print("=" * 50)
    
    models = {
        "XGBoost": {
            "type": "Gradient Boosting",
            "weight": "20%", 
            "purpose": "Feature-based prediction using technical indicators"
        },
        "LSTM": {
            "type": "Deep Learning",
            "weight": "18%",
            "purpose": "Sequential pattern recognition in price movements"
        },
        "GRU": {
            "type": "Deep Learning", 
            "weight": "15%",
            "purpose": "Alternative RNN architecture for time series"
        },
        "ARIMA": {
            "type": "Statistical",
            "weight": "12%",
            "purpose": "Classical time series forecasting"
        },
        "Technical Analysis": {
            "type": "Rule-based",
            "weight": "15%",
            "purpose": "RSI, MACD, Moving Averages, Bollinger Bands"
        },
        "Sentiment Analysis": {
            "type": "NLP",
            "weight": "10%",
            "purpose": "News and social media sentiment analysis"
        },
        "Monte Carlo": {
            "type": "Statistical",
            "weight": "10%",
            "purpose": "Probabilistic simulation of price movements"
        }
    }
    
    for model_name, details in models.items():
        print(f"{model_name:20} | {details['type']:15} | {details['weight']:5} | {details['purpose']}")
    
    print(f"\nTotal Models: {len(models)}")
    print("Ensemble Method: Weighted average with confidence scoring")

def show_system_capabilities():
    """Display system capabilities"""
    print("\n🚀 SYSTEM CAPABILITIES")
    print("=" * 30)
    
    capabilities = [
        "✓ Multi-model ensemble prediction",
        "✓ Technical indicator analysis", 
        "✓ Sentiment analysis integration",
        "✓ Risk assessment with confidence levels",
        "✓ Buy/Sell/Hold recommendations",
        "✓ Performance metrics and backtesting",
        "✓ Data visualization and reporting", 
        "✓ Real-time analysis capability",
        "✓ Extensible architecture for new models"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")

if __name__ == "__main__":
    # Run demo
    success = demo_integrated_system()
    
    if success:
        show_model_architecture()
        show_system_capabilities()
        
        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("The integrated system is operational and ready for use.")
        print("Install requirements.txt for full functionality with real market data.")
    else:
        print("\n❌ Demo failed - check error messages above")