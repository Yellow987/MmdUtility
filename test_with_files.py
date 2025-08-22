#!/usr/bin/env python3
"""
Test script for the bone position extractor module with actual PMX and VMD files.
Tests loading functionality with real files in the test directory.
"""

import os
import sys

def test_with_real_files():
    """Test loading actual PMX and VMD files."""
    print("Testing with real PMX and VMD files...")
    
    # Define test file paths
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    # Check if files exist
    if not os.path.exists(pmx_path):
        print(f"✗ PMX test file not found: {pmx_path}")
        return False
    
    if not os.path.exists(vmd_path):
        print(f"✗ VMD test file not found: {vmd_path}")
        return False
    
    print(f"Found PMX file: {pmx_path}")
    print(f"Found VMD file: {vmd_path}")
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        # Test PMX loading
        print("\nTesting PMX loading...")
        try:
            extractor.load_pmx(pmx_path)
            print("✓ PMX file loaded successfully")
            
            # Get and display model info
            model_info = extractor.get_model_info()
            print(f"  Model name: {model_info.get('model_name', 'Unknown')}")
            print(f"  English name: {model_info.get('english_name', 'Unknown')}")
            print(f"  Bones: {model_info.get('bone_count', 0)}")
            print(f"  Vertices: {model_info.get('vertex_count', 0)}")
            print(f"  Materials: {model_info.get('material_count', 0)}")
            
        except Exception as e:
            print(f"✗ Failed to load PMX file: {e}")
            return False
        
        # Test VMD loading
        print("\nTesting VMD loading...")
        try:
            extractor.load_vmd(vmd_path)
            print("✓ VMD file loaded successfully")
            
            # Get and display motion info
            motion_info = extractor.get_motion_info()
            print(f"  Motion model name: {motion_info.get('model_name', 'Unknown')}")
            print(f"  Bone motions: {motion_info.get('bone_motion_count', 0)}")
            print(f"  Morph count: {motion_info.get('morph_count', 0)}")
            print(f"  Camera frames: {motion_info.get('camera_frame_count', 0)}")
            print(f"  Max frame: {motion_info.get('max_frame', 0)}")
            
        except Exception as e:
            print(f"✗ Failed to load VMD file: {e}")
            return False
        
        # Test the main extraction function (even though it's just a placeholder)
        print("\nTesting extract_bone_positions function...")
        try:
            positions = extractor.extract_bone_positions(pmx_path, vmd_path)
            print(f"✓ extract_bone_positions executed successfully")
            print(f"  Returned {len(positions)} frame(s)")
            if positions:
                sample_frame = positions[0]
                print(f"  Sample frame contains {len(sample_frame)} bones")
                # Show first few bone names as sample
                bone_names = list(sample_frame.keys())[:5]
                print(f"  Sample bone names: {bone_names}")
            
        except Exception as e:
            print(f"✗ extract_bone_positions failed: {e}")
            return False
        
        print("\n✓ All file loading tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions with real files."""
    print("\nTesting convenience functions...")
    
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    if not os.path.exists(pmx_path) or not os.path.exists(vmd_path):
        print("ℹ Test files not available, skipping convenience function tests")
        return True
    
    try:
        import bone_position_extractor
        
        # Test get_file_info convenience function
        try:
            model_info, motion_info = bone_position_extractor.get_file_info(pmx_path, vmd_path)
            print("✓ get_file_info convenience function works")
            print(f"  Model: {model_info.get('model_name', 'Unknown')}")
            print(f"  Motion: {motion_info.get('model_name', 'Unknown')}")
        except Exception as e:
            print(f"✗ get_file_info failed: {e}")
            return False
        
        # Test extract_bone_positions convenience function
        try:
            positions = bone_position_extractor.extract_bone_positions(pmx_path, vmd_path)
            print("✓ extract_bone_positions convenience function works")
            print(f"  Returned {len(positions)} frame(s)")
        except Exception as e:
            print(f"✗ extract_bone_positions convenience function failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience function test failed: {e}")
        return False

def main():
    """Run all tests with real files."""
    print("=" * 60)
    print("Bone Position Extractor - Real File Tests")
    print("=" * 60)
    
    tests = [
        ("Real File Loading Test", test_with_real_files),
        ("Convenience Functions Test", test_convenience_functions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
        
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The module successfully loads PMX and VMD files.")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)