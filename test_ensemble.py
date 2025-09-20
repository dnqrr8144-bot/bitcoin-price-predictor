#!/usr/bin/env python
"""
Test script for the Single-File Hybrid Ensemble system
Demonstrates various usage scenarios
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

def main():
    """Run all tests"""
    print("🧪 Testing Single-File Hybrid Ensemble System")
    print("=" * 50)
    
    tests = [
        ("Basic Analysis", test_basic_analysis),
        ("Custom Weights", test_custom_weights),
        ("JSON Export", test_json_export),
        ("Debug Mode", test_debug_mode),
        ("Fundamental Override", test_fundamental_override)
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
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The ensemble system is working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the output above.")
    
    # Cleanup
    if os.path.exists("test_output.json"):
        os.remove("test_output.json")

if __name__ == "__main__":
    main()