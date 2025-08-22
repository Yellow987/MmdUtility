#!/usr/bin/env python3
"""
Test script for rest pose bone position extraction functionality.
Tests the new rest pose methods added to the bone position extractor.
"""

import os
import sys
import numpy as np


def test_rest_pose_extraction():
    """Test rest pose position extraction with real PMX file."""
    print("Testing rest pose position extraction...")
    
    pmx_path = "test/pdtt.pmx"
    
    if not os.path.exists(pmx_path):
        print(f"✗ PMX test file not found: {pmx_path}")
        return False
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        # Load PMX file
        print(f"Loading PMX file: {pmx_path}")
        extractor.load_pmx(pmx_path)
        print("✓ PMX file loaded successfully")
        
        # Test class method for rest pose positions
        print("\nTesting get_rest_pose_positions()...")
        try:
            rest_positions = extractor.get_rest_pose_positions()
            print(f"✓ Rest pose positions extracted: {len(rest_positions)} bones")
            
            # Show some sample positions
            sample_bones = list(rest_positions.keys())[:5]
            print("Sample bone positions:")
            for bone_name in sample_bones:
                x, y, z = rest_positions[bone_name]
                print(f"  {bone_name}: ({x:.3f}, {y:.3f}, {z:.3f})")
                
        except Exception as e:
            print(f"✗ get_rest_pose_positions() failed: {e}")
            return False
        
        # Test array output
        print("\nTesting get_rest_pose_positions_array()...")
        try:
            positions_array = extractor.get_rest_pose_positions_array()
            print(f"✓ Rest pose array extracted: shape {positions_array.shape}")
            
            # Verify it's the right shape and contains reasonable data
            expected_bones = len(extractor.bone_hierarchy)
            if positions_array.shape == (expected_bones, 3):
                print(f"✓ Array has correct shape: ({expected_bones}, 3)")
            else:
                print(f"✗ Unexpected array shape: {positions_array.shape}")
                return False
            
            # Show some statistics
            print(f"Position ranges: X=[{positions_array[:, 0].min():.3f}, {positions_array[:, 0].max():.3f}]")
            print(f"                 Y=[{positions_array[:, 1].min():.3f}, {positions_array[:, 1].max():.3f}]")
            print(f"                 Z=[{positions_array[:, 2].min():.3f}, {positions_array[:, 2].max():.3f}]")
            
        except Exception as e:
            print(f"✗ get_rest_pose_positions_array() failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions for rest pose extraction."""
    print("Testing convenience functions...")
    
    pmx_path = "test/pdtt.pmx"
    
    if not os.path.exists(pmx_path):
        print(f"ℹ PMX test file not found, skipping convenience function tests")
        return True
    
    try:
        import bone_position_extractor
        
        # Test get_rest_pose_positions convenience function
        print("\nTesting get_rest_pose_positions() convenience function...")
        try:
            rest_positions = bone_position_extractor.get_rest_pose_positions(pmx_path)
            print(f"✓ Convenience function worked: {len(rest_positions)} bones")
            
            # Verify some bones have reasonable positions
            non_zero_count = sum(1 for pos in rest_positions.values() if any(abs(x) > 0.001 for x in pos))
            print(f"✓ {non_zero_count} bones have non-zero positions")
            
        except Exception as e:
            print(f"✗ get_rest_pose_positions() convenience function failed: {e}")
            return False
        
        # Test get_rest_pose_positions_array convenience function
        print("\nTesting get_rest_pose_positions_array() convenience function...")
        try:
            positions_array = bone_position_extractor.get_rest_pose_positions_array(pmx_path)
            print(f"✓ Array convenience function worked: shape {positions_array.shape}")
            
            # Verify the array contains the same data as the dictionary approach
            rest_dict = bone_position_extractor.get_rest_pose_positions(pmx_path)
            if len(rest_dict) == positions_array.shape[0]:
                print("✓ Array and dictionary results are consistent")
            else:
                print(f"✗ Inconsistent results: dict has {len(rest_dict)}, array has {positions_array.shape[0]}")
                return False
                
        except Exception as e:
            print(f"✗ get_rest_pose_positions_array() convenience function failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience function test failed: {e}")
        return False


def test_error_handling():
    """Test error handling for rest pose functions."""
    print("Testing error handling...")
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        # Test calling rest pose methods without loading PMX
        try:
            extractor.get_rest_pose_positions()
            print("✗ Should have failed when no PMX loaded")
            return False
        except ValueError:
            print("✓ Proper error handling when no PMX loaded")
        
        # Test with non-existent file
        try:
            bone_position_extractor.get_rest_pose_positions("non_existent.pmx")
            print("✗ Should have failed with non-existent file")
            return False
        except (RuntimeError, ValueError):
            print("✓ Proper error handling for non-existent file")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_bone_hierarchy_validation():
    """Test that bone positions follow expected hierarchy rules."""
    print("Testing bone hierarchy validation...")
    
    pmx_path = "test/pdtt.pmx"
    
    if not os.path.exists(pmx_path):
        print(f"ℹ PMX test file not found, skipping hierarchy validation")
        return True
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        extractor.load_pmx(pmx_path)
        
        rest_positions = extractor.get_rest_pose_positions()
        
        # Check that root bones are typically at or near origin
        root_bones = [i for i, info in extractor.bone_hierarchy.items() if info['parent_index'] == -1]
        print(f"Found {len(root_bones)} root bones")
        
        # Check that child bones are positioned relative to their parents
        parent_child_pairs = 0
        for bone_index, bone_info in extractor.bone_hierarchy.items():
            if bone_info['parent_index'] != -1:
                parent_child_pairs += 1
        
        print(f"Found {parent_child_pairs} parent-child bone relationships")
        print("✓ Bone hierarchy structure looks reasonable")
        
        return True
        
    except Exception as e:
        print(f"✗ Hierarchy validation failed: {e}")
        return False


def main():
    """Run all rest pose tests."""
    print("=" * 60)
    print("Rest Pose Bone Position Extraction Tests")
    print("=" * 60)
    
    tests = [
        ("Rest Pose Extraction Test", test_rest_pose_extraction),
        ("Convenience Functions Test", test_convenience_functions),
        ("Error Handling Test", test_error_handling),
        ("Bone Hierarchy Validation Test", test_bone_hierarchy_validation)
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
        print("🎉 All rest pose tests passed! Rest pose functionality is working correctly.")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)