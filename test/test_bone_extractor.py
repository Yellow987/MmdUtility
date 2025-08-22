#!/usr/bin/env python3
"""
Test script for the bone position extractor module.

This script tests the basic functionality of loading PMX and VMD files.
"""

import os
import sys
from bone_position_extractor import BonePositionExtractor, extract_bone_positions, get_file_info


def test_basic_functionality():
    """Test basic module import and class instantiation."""
    print("Testing basic functionality...")
    
    try:
        extractor = BonePositionExtractor()
        print("✓ BonePositionExtractor class instantiated successfully")
    except Exception as e:
        print(f"✗ Failed to instantiate BonePositionExtractor: {e}")
        return False
    
    return True


def test_file_loading(pmx_path=None, vmd_path=None):
    """Test PMX and VMD file loading if files are available."""
    print("Testing file loading...")
    
    if not pmx_path or not vmd_path:
        print("ℹ No test files provided, skipping file loading tests")
        return True
    
    if not os.path.exists(pmx_path):
        print(f"ℹ PMX test file not found: {pmx_path}")
        return True
    
    if not os.path.exists(vmd_path):
        print(f"ℹ VMD test file not found: {vmd_path}")
        return True
    
    extractor = BonePositionExtractor()
    
    # Test PMX loading
    try:
        extractor.load_pmx(pmx_path)
        print("✓ PMX file loaded successfully")
        
        model_info = extractor.get_model_info()
        print(f"  Model: {model_info.get('model_name', 'Unknown')}")
        print(f"  Bones: {model_info.get('bone_count', 0)}")
        
    except Exception as e:
        print(f"✗ Failed to load PMX file: {e}")
        return False
    
    # Test VMD loading
    try:
        extractor.load_vmd(vmd_path)
        print("✓ VMD file loaded successfully")
        
        motion_info = extractor.get_motion_info()
        print(f"  Motion model: {motion_info.get('model_name', 'Unknown')}")
        print(f"  Bone motions: {motion_info.get('bone_motion_count', 0)}")
        print(f"  Max frame: {motion_info.get('max_frame', 0)}")
        
    except Exception as e:
        print(f"✗ Failed to load VMD file: {e}")
        return False
    
    return True


def test_convenience_functions():
    """Test convenience functions with dummy paths."""
    print("Testing convenience functions...")
    
    try:
        # Test with non-existent files to check error handling
        try:
            positions = extract_bone_positions("dummy.pmx", "dummy.vmd")
            print("✗ Should have raised an error with dummy files")
            return False
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            print("✓ Proper error handling for non-existent files")
    
    except Exception as e:
        print(f"✗ Unexpected error in convenience function test: {e}")
        return False
    
    return True


def run_all_tests(pmx_path=None, vmd_path=None):
    """Run all tests."""
    print("=" * 50)
    print("Bone Position Extractor Module Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic functionality
    if test_basic_functionality():
        tests_passed += 1
    print()
    
    # Test 2: File loading (if files available)
    if test_file_loading(pmx_path, vmd_path):
        tests_passed += 1
    print()
    
    # Test 3: Convenience functions
    if test_convenience_functions():
        tests_passed += 1
    print()
    
    # Results
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


def main():
    """Main test function."""
    # Check for command line arguments for test files
    pmx_path = None
    vmd_path = None
    
    if len(sys.argv) >= 3:
        pmx_path = sys.argv[1]
        vmd_path = sys.argv[2]
        print(f"Using test files: PMX={pmx_path}, VMD={vmd_path}")
    else:
        print("Usage: python test_bone_extractor.py [pmx_file] [vmd_file]")
        print("Running tests without actual files...")
    
    success = run_all_tests(pmx_path, vmd_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()