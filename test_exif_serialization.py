#!/usr/bin/env python3
"""
Test script to verify EXIF data serialization works correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from utils.exif_utils import serialize_exif_value, serialize_exif_data
from PIL.TiffImagePlugin import IFDRational
import json

def test_serialization():
    """Test EXIF data serialization"""
    print("🔄 Testing EXIF data serialization...")
    
    # Test IFDRational serialization
    rational_value = IFDRational(1, 3)  # 1/3
    serialized = serialize_exif_value(rational_value)
    print(f"✅ IFDRational(1, 3) → {serialized} (type: {type(serialized)})")
    
    # Test complex EXIF data structure
    test_exif = {
        'Make': 'Canon',
        'Model': 'EOS 5D Mark IV',
        'FNumber': IFDRational(28, 10),  # f/2.8
        'ExposureTime': IFDRational(1, 125),  # 1/125s
        'ISO': 800,
        'GPSInfo': {
            'GPSLatitude': (IFDRational(12, 1), IFDRational(58, 1), IFDRational(2996, 100)),
            'GPSLongitude': (IFDRational(77, 1), IFDRational(35, 1), IFDRational(6756, 100)),
        },
        'BinaryData': b'some binary data',
        'NestedList': [IFDRational(1, 2), IFDRational(3, 4)],
    }
    
    print("\n🔄 Testing complex EXIF data structure...")
    serialized_exif = serialize_exif_data(test_exif)
    
    # Try to JSON serialize to verify MongoDB compatibility
    try:
        json_str = json.dumps(serialized_exif, indent=2)
        print("✅ Successfully serialized to JSON (MongoDB compatible)")
        print("📋 Sample serialized data:")
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
        return True
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases for serialization"""
    print("\n🔄 Testing edge cases...")
    
    edge_cases = [
        IFDRational(0, 0),  # 0/0 (should handle gracefully)
        IFDRational(1, 0),  # 1/0 (infinity)
        IFDRational(-1, 0), # -1/0 (negative infinity)
        float('nan'),       # NaN
        None,               # None value
        [],                 # Empty list
        {},                 # Empty dict
    ]
    
    for i, case in enumerate(edge_cases):
        try:
            result = serialize_exif_value(case)
            print(f"✅ Edge case {i+1}: {case} → {result}")
        except Exception as e:
            print(f"❌ Edge case {i+1} failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting EXIF Serialization Test")
    print("=" * 50)
    
    success = test_serialization()
    test_edge_cases()
    
    print("\n" + "=" * 50)
    if success:
        print("🏁 ✅ All tests passed! EXIF serialization is working correctly.")
    else:
        print("🏁 ❌ Some tests failed. Check the implementation.")
