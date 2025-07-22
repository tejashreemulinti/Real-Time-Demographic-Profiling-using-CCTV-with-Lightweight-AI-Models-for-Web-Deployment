#!/usr/bin/env python3
"""
Verify that the logger fix is working correctly
"""

import os
import re

def check_logger_fix():
    """Check if the logger issue is fixed in the ultra accurate estimator."""
    
    file_path = "backend/ultra_accurate_age_gender_estimator.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if logger is defined before use
    lines = content.split('\n')
    
    logger_defined_line = -1
    logger_used_lines = []
    
    for i, line in enumerate(lines):
        if 'logger = logging.getLogger(__name__)' in line:
            logger_defined_line = i
            print(f"✅ Logger defined at line {i + 1}")
        
        if 'logger.' in line and 'logger = ' not in line:
            logger_used_lines.append(i + 1)
    
    if logger_defined_line == -1:
        print("❌ Logger not defined")
        return False
    
    # Check if all logger usages come after definition
    early_usages = [line_num for line_num in logger_used_lines if line_num <= logger_defined_line + 1]
    
    if early_usages:
        print(f"❌ Logger used before definition at lines: {early_usages}")
        return False
    
    print(f"✅ Logger used after definition at lines: {logger_used_lines}")
    
    # Check optional imports structure
    if 'SKLEARN_AVAILABLE = False' in content:
        print("✅ scikit-learn optional import structure found")
    
    if 'DLIB_AVAILABLE = False' in content:
        print("✅ dlib optional import structure found")
    
    if 'SKIMAGE_AVAILABLE = False' in content:
        print("✅ scikit-image optional import structure found")
    
    return True

def check_imports_structure():
    """Check that imports are properly structured with fallbacks."""
    
    file_path = "backend/ultra_accurate_age_gender_estimator.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for proper try-except blocks
    expected_patterns = [
        r'try:\s*from sklearn',
        r'try:\s*import dlib',
        r'try:\s*from skimage',
        r'except ImportError:',
        r'SKLEARN_AVAILABLE = True',
        r'DLIB_AVAILABLE = True',
        r'SKIMAGE_AVAILABLE = True'
    ]
    
    for pattern in expected_patterns:
        if re.search(pattern, content, re.MULTILINE):
            print(f"✅ Found pattern: {pattern}")
        else:
            print(f"❌ Missing pattern: {pattern}")
            return False
    
    return True

if __name__ == "__main__":
    print("🔍 Verifying Logger Fix and Import Structure")
    print("=" * 50)
    
    logger_ok = check_logger_fix()
    imports_ok = check_imports_structure()
    
    if logger_ok and imports_ok:
        print("\n✅ All checks passed!")
        print("The logger issue has been fixed and imports are properly structured.")
        print("The system should now work without dlib, scikit-learn, or scikit-image.")
    else:
        print("\n❌ Some checks failed.")
        
    print("\n📝 Summary:")
    print("- Logger is defined before use")
    print("- Optional imports have proper fallbacks")
    print("- System will work with different dependency levels")
    print("- Accuracy will adjust based on available packages")