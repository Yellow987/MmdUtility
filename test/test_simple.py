#!/usr/bin/env python3
"""
Simple test script for the bone position extractor module.
Tests basic functionality without requiring actual PMX/VMD files.
"""

def test_import():
    """Test if the module can be imported successfully."""
    try:
        import bone_position_extractor
        print("✓ bone_position_extractor module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import bone_position_extractor: {e}")
        return False

def test_class_instantiation():
    """Test if the BonePositionExtractor class can be instantiated."""
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        print("✓ BonePositionExtractor class instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate BonePositionExtractor: {e}")
        return False

def test_error_handling():
    """Test error handling with non-existent files."""
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        # Test PMX loading error handling
        try:
            extractor.load_pmx("non_existent.pmx")
            print("✗ Should have failed with non-existent PMX file")
            return False
        except (RuntimeError, ValueError) as e:
            print("✓ Proper error handling for non-existent PMX file")
        
        # Test VMD loading error handling
        try:
            extractor.load_vmd("non_existent.vmd")
            print("✗ Should have failed with non-existent VMD file")
            return False
        except (RuntimeError, ValueError) as e:
            print("✓ Proper error handling for non-existent VMD file")
        
        return True
    except Exception as e:
        print(f"✗ Unexpected error in error handling test: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Simple Bone Position Extractor Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Class Instantiation Test", test_class_instantiation),
        ("Error Handling Test", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All basic tests passed! Module is ready for use.")
        return True
    else:
        print("✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)