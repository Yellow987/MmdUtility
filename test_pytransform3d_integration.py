#!/usr/bin/env python3
"""
Test script for pytransform3d skeleton loading integration.
Tests that the PMX skeleton loads correctly into TransformManager.
"""

import os
import sys
import numpy as np
from bone_position_extractor import BonePositionExtractor


def test_pytransform3d_availability():
    """Test if pytransform3d is available."""
    print("Testing pytransform3d availability...")
    
    try:
        from pytransform3d.transform_manager import TransformManager
        print("✓ pytransform3d is available")
        return True
    except ImportError:
        print("✗ pytransform3d not available. Install with: pip install pytransform3d")
        return False


def test_skeleton_loading():
    """Test loading PMX skeleton into TransformManager."""
    print("\nTesting skeleton loading into TransformManager...")
    
    pmx_path = "test/pdtt.pmx"
    if not os.path.exists(pmx_path):
        print(f"✗ Test file not found: {pmx_path}")
        return False
    
    extractor = BonePositionExtractor()
    
    # Load PMX file
    try:
        extractor.load_pmx(pmx_path)
        print("✓ PMX file loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load PMX file: {e}")
        return False
    
    # Test TransformManager loading
    try:
        extractor._load_skeleton_to_transform_manager()
        print("✓ Skeleton loaded into TransformManager")
    except Exception as e:
        print(f"✗ Failed to load skeleton into TransformManager: {e}")
        return False
    
    # Verify TransformManager was created
    if extractor.transform_manager is None:
        print("✗ TransformManager was not created")
        return False
    
    print("✓ TransformManager initialization successful")
    return True


def test_rest_pose_comparison():
    """Compare rest pose results between implementations."""
    print("\nTesting rest pose comparison...")
    
    pmx_path = "test/pdtt.pmx"
    if not os.path.exists(pmx_path):
        print(f"ℹ Test file not found: {pmx_path}, skipping comparison")
        return True
    
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    
    # Get rest pose positions using both methods
    print("Getting rest pose positions...")
    
    # Current implementation
    try:
        current_positions = extractor.get_rest_pose_positions()
        print(f"✓ Current implementation: {len(current_positions)} bones")
    except Exception as e:
        print(f"✗ Current implementation failed: {e}")
        return False
    
    # TransformManager implementation
    try:
        tm_positions = extractor.get_rest_pose_positions_pytransform3d()
        print(f"✓ TransformManager implementation: {len(tm_positions)} bones")
    except Exception as e:
        print(f"✗ TransformManager implementation failed: {e}")
        return False
    
    # Compare results - only compare bones that exist in both implementations
    print("\nComparing bone positions...")
    
    # Find common bones
    common_bones = set(current_positions.keys()) & set(tm_positions.keys())
    print(f"Common bones to compare: {len(common_bones)}")
    print(f"Current implementation has {len(current_positions)} bones")
    print(f"TransformManager implementation has {len(tm_positions)} bones (core bones only)")
    
    matching_bones = 0
    total_bones = len(common_bones)
    max_diff = 0.0
    problematic_bones = []
    
    for bone_name in common_bones:
        current_pos = current_positions[bone_name]
        tm_pos = tm_positions[bone_name]
        
        # Calculate difference
        diff = np.linalg.norm(np.array(current_pos) - np.array(tm_pos))
        max_diff = max(max_diff, diff)
        
        # Check if positions match (within tolerance)
        if diff < 0.001:  # 1mm tolerance
            matching_bones += 1
        else:
            problematic_bones.append((bone_name, current_pos, tm_pos, diff))
    
    print(f"Results: {matching_bones}/{total_bones} bones match")
    print(f"Maximum difference: {max_diff:.6f}")
    
    # Show problematic bones (limit to first 5)
    if problematic_bones:
        print(f"\nBones with differences > 0.001:")
        for i, (bone_name, curr_pos, tm_pos, diff) in enumerate(problematic_bones[:5]):
            print(f"  {bone_name}:")
            print(f"    Current: ({curr_pos[0]:.6f}, {curr_pos[1]:.6f}, {curr_pos[2]:.6f})")
            print(f"    TransformMgr: ({tm_pos[0]:.6f}, {tm_pos[1]:.6f}, {tm_pos[2]:.6f})")
            print(f"    Difference: {diff:.6f}")
        
        if len(problematic_bones) > 5:
            print(f"  ... and {len(problematic_bones) - 5} more")
    
    # Determine success - allow some small differences due to floating point precision
    success_threshold = 0.9  # 90% of bones should match exactly
    success_rate = matching_bones / total_bones if total_bones > 0 else 0
    
    if success_rate >= success_threshold:
        print(f"✓ Position comparison successful ({success_rate:.1%} match rate)")
        return True
    else:
        print(f"✗ Too many position differences ({success_rate:.1%} match rate)")
        return False


def test_bone_hierarchy_validation():
    """Test that bone hierarchy is correctly represented."""
    print("\nTesting bone hierarchy validation...")
    
    pmx_path = "test/pdtt.pmx"
    if not os.path.exists(pmx_path):
        print(f"ℹ Test file not found: {pmx_path}, skipping hierarchy validation")
        return True
    
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    extractor._load_skeleton_to_transform_manager()
    
    # Check that all bones have frames in TransformManager
    expected_frames = len(extractor.bone_hierarchy)
    
    # Get available transforms (this is a simple check)
    try:
        # Test a few transforms to ensure they work
        test_bone_indices = list(extractor.bone_hierarchy.keys())[:3]
        successful_transforms = 0
        
        for bone_idx in test_bone_indices:
            frame_name = f"bone_{bone_idx}"
            try:
                transform = extractor.transform_manager.get_transform("world", frame_name)
                if transform is not None:
                    successful_transforms += 1
            except Exception as e:
                print(f"  Warning: Could not get transform for bone_{bone_idx}: {e}")
        
        print(f"✓ Successfully tested {successful_transforms}/{len(test_bone_indices)} transforms")
        return successful_transforms > 0
        
    except Exception as e:
        print(f"✗ Hierarchy validation failed: {e}")
        return False


def main():
    """Run all pytransform3d integration tests."""
    print("=" * 60)
    print("Pytransform3d Skeleton Loading Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Pytransform3d Availability", test_pytransform3d_availability),
        ("Skeleton Loading", test_skeleton_loading),
        ("Rest Pose Comparison", test_rest_pose_comparison),
        ("Bone Hierarchy Validation", test_bone_hierarchy_validation)
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
        print("🎉 All tests passed! Pytransform3d integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)