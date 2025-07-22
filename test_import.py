#!/usr/bin/env python3
"""
Simple test to verify ultra accurate estimator imports without dlib
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    print("Testing import of ultra accurate estimator...")
    
    try:
        from backend.ultra_accurate_age_gender_estimator import UltraAccurateAgeGenderEstimator
        print("✅ Successfully imported UltraAccurateAgeGenderEstimator")
        
        # Try to create an instance
        estimator = UltraAccurateAgeGenderEstimator()
        print("✅ Successfully created estimator instance")
        
        print(f"📊 Age groups available: {len(estimator.age_groups)}")
        print(f"🎯 Sample age groups: {estimator.age_groups[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 Testing Ultra Accurate Age Gender Estimator Import")
    print("=" * 60)
    
    success = test_import()
    
    if success:
        print("\n✅ All tests passed! The system should work without dlib.")
    else:
        print("\n❌ Tests failed. There may be an issue with the imports.")
        sys.exit(1)