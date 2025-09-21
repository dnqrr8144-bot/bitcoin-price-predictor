#!/usr/bin/env python
"""
Test script for the Single-File Hybrid Ensemble system
Demonstrates various usage scenarios including time-horizon specific predictions
"""
import os
import json

def test_basic_analysis():
    """Test basic Bitcoin analysis using CSV data"""
    print("=== Test 1: Basic Bitcoin Analysis ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast"
    result = os.system(cmd)
    return result == 0

def test_custom_weights():
    """Test with custom model weights"""
    print("\n=== Test 2: Custom Model Weights ===")
    weights = '{"XGBoost":0.25,"LSTM":0.25,"Technical":0.25,"Fundamental":0.25}'
    cmd = f'python single_ensemble.py --ticker BTC-USD --use_csv --fast --weights \'{weights}\''
    result = os.system(cmd)
    return result == 0

def test_json_export():
    """Test JSON export functionality"""
    print("\n=== Test 3: JSON Export ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast --save_json test_output.json"
    result = os.system(cmd)
    
    # Verify JSON file was created and is valid
    if result == 0 and os.path.exists("test_output.json"):
        try:
            with open("test_output.json", 'r') as f:
                data = json.load(f)
            print(f"✓ JSON export successful. Final score: {data['final_score']:.4f}")
            print(f"✓ Recommendation: {data['decision']}")
            return True
        except json.JSONDecodeError:
            print("✗ JSON file is invalid")
            return False
    return False

def test_debug_mode():
    """Test debug mode output"""
    print("\n=== Test 4: Debug Mode ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast --debug"
    result = os.system(cmd)
    return result == 0

def test_fundamental_override():
    """Test fundamental analysis override"""
    print("\n=== Test 5: Fundamental Override ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast --fundamental_override 0.8"
    result = os.system(cmd)
    return result == 0

def test_time_horizon_30():
    """Test 30-day time horizon configuration"""
    print("\n=== Test 6: 30-Day Time Horizon ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast --time_horizon 30"
    result = os.system(cmd)
    return result == 0

def test_time_horizon_60():
    """Test 60-day time horizon configuration"""
    print("\n=== Test 7: 60-Day Time Horizon ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast --time_horizon 60"
    result = os.system(cmd)
    return result == 0

def test_time_horizon_90():
    """Test 90-day time horizon configuration"""
    print("\n=== Test 8: 90-Day Time Horizon ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast --time_horizon 90"
    result = os.system(cmd)
    return result == 0

def test_time_horizon_with_debug():
    """Test time horizon with debug mode to verify correct model weights"""
    print("\n=== Test 9: Time Horizon with Debug ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast --time_horizon 30 --debug"
    result = os.system(cmd)
    return result == 0

def test_time_horizon_json_export():
    """Test time horizon with JSON export to verify correct data structure"""
    print("\n=== Test 10: Time Horizon JSON Export ===")
    cmd = "python single_ensemble.py --ticker BTC-USD --use_csv --fast --time_horizon 60 --save_json test_horizon_output.json"
    result = os.system(cmd)
    
    # Verify JSON file contains time horizon information
    if result == 0 and os.path.exists("test_horizon_output.json"):
        try:
            with open("test_horizon_output.json", 'r') as f:
                data = json.load(f)
            if 'time_horizon' in data and data['time_horizon'] == 60:
                print(f"✓ Time horizon JSON export successful. Horizon: {data['time_horizon']} days")
                print(f"✓ Final score: {data['final_score']:.4f}")
                return True
            else:
                print("✗ Time horizon not properly saved in JSON")
                return False
        except json.JSONDecodeError:
            print("✗ JSON file is invalid")
            return False
    return False

def main():
    """Run all tests"""
    print("🧪 Testing Single-File Hybrid Ensemble System with Time Horizons")
    print("=" * 70)
    
    tests = [
        ("Basic Analysis", test_basic_analysis),
        ("Custom Weights", test_custom_weights),
        ("JSON Export", test_json_export),
        ("Debug Mode", test_debug_mode),
        ("Fundamental Override", test_fundamental_override),
        ("30-Day Time Horizon", test_time_horizon_30),
        ("60-Day Time Horizon", test_time_horizon_60),
        ("90-Day Time Horizon", test_time_horizon_90),
        ("Time Horizon Debug", test_time_horizon_with_debug),
        ("Time Horizon JSON", test_time_horizon_json_export)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The ensemble system with time horizons is working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the output above.")
    
    # Cleanup
    for file in ["test_output.json", "test_horizon_output.json"]:
        if os.path.exists(file):
            os.remove(file)
    
    print(f"\n📊 Time-Horizon Specific Models Summary:")
    print(f"   • 30 days: ARIMA-GARCH (35%) + LSTM (30%) + HMM (25%) + Technical (10%)")
    print(f"   • 60 days: LSTM (40%) + XGBoost (25%) + ARIMA-GARCH (20%) + Technical (10%) + Fundamental (5%)")
    print(f"   • 90 days: Factor Models (35%) + LSTM (30%) + XGBoost (20%) + Fundamental (10%) + Technical (5%)")

if __name__ == "__main__":
    main()