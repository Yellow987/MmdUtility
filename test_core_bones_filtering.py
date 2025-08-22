#!/usr/bin/env python3
"""
Test script to verify core bone filtering is working.
"""

import os
from bone_position_extractor import BonePositionExtractor


def test_core_bone_filtering():
    """Test that only core bones are being processed."""
    pmx_path = "test/pdtt.pmx"
    if not os.path.exists(pmx_path):
        print(f"Test file not found: {pmx_path}")
        return
    
    print("Testing core bone filtering...")
    
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    
    print(f"\nTotal bones in PMX: {len(extractor.pmx_model.bones)}")
    
    # Get positions using current implementation (all bones)
    current_positions = extractor.get_rest_pose_positions()
    print(f"Current implementation bones: {len(current_positions)}")
    
    # Get positions using TransformManager (core bones only)
    tm_positions = extractor.get_rest_pose_positions_pytransform3d()
    print(f"TransformManager core bones: {len(tm_positions)}")
    
    print(f"\nCore bones found:")
    for bone_name in sorted(tm_positions.keys()):
        if bone_name == "":
            print(f"  '' (root bone)")
        else:
            print(f"  '{bone_name}'")
    
    # Show some position comparisons for core bones
    print(f"\nPosition comparisons for first 5 core bones:")
    for i, bone_name in enumerate(sorted(tm_positions.keys())[:5]):
        if bone_name in current_positions:
            curr_pos = current_positions[bone_name]
            tm_pos = tm_positions[bone_name]
            print(f"  {bone_name if bone_name else '(root)'}:")
            print(f"    Current:      ({curr_pos[0]:.6f}, {curr_pos[1]:.6f}, {curr_pos[2]:.6f})")
            print(f"    TransformMgr: ({tm_pos[0]:.6f}, {tm_pos[1]:.6f}, {tm_pos[2]:.6f})")


if __name__ == "__main__":
    test_core_bone_filtering()