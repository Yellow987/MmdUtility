#!/usr/bin/env python3
"""
Debug script to identify bone names for the problematic bones.
"""

import os
import sys
import numpy as np
from bone_position_extractor import BonePositionExtractor


def debug_bone_names():
    """Debug bone names and identify the problematic bone."""
    print("Debugging bone names...")
    
    pmx_path = "test/pdtt.pmx"
    if not os.path.exists(pmx_path):
        print(f"Test file not found: {pmx_path}")
        return False
    
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    
    # Get rest pose positions using both methods
    current_positions = extractor.get_rest_pose_positions()
    tm_positions = extractor.get_rest_pose_positions_pytransform3d()
    
    print(f"\nTotal bones: {len(current_positions)}")
    print("\nFirst 10 bone names from current implementation:")
    for i, bone_name in enumerate(list(current_positions.keys())[:10]):
        print(f"  {i}: '{bone_name}' (repr: {repr(bone_name)})")
    
    print("\nFirst 10 bone names from TransformManager implementation:")
    for i, bone_name in enumerate(list(tm_positions.keys())[:10]):
        print(f"  {i}: '{bone_name}' (repr: {repr(bone_name)})")
    
    # Find the problematic bone
    print("\nLooking for bones with significant differences...")
    
    for bone_name, current_pos in current_positions.items():
        if bone_name in tm_positions:
            tm_pos = tm_positions[bone_name]
            diff = np.linalg.norm(np.array(current_pos) - np.array(tm_pos))
            
            if diff > 10.0:  # Look for the big difference
                print(f"\nFound problematic bone:")
                print(f"  Name: '{bone_name}'")
                print(f"  Name repr: {repr(bone_name)}")
                print(f"  Current pos: {current_pos}")
                print(f"  TransformMgr pos: {tm_pos}")
                print(f"  Difference: {diff:.6f}")
                
                # Check if this matches the ID from the original output
                if "#3997a7ac" in bone_name or bone_name == "#3997a7ac":
                    print("  ✓ This matches the problematic bone from the original output!")
                
                # Find the bone index and get more info
                if bone_name in extractor.bone_name_to_index:
                    bone_idx = extractor.bone_name_to_index[bone_name]
                    print(f"  Bone index: {bone_idx}")
                    
                    if bone_idx in extractor.bone_hierarchy:
                        bone_info = extractor.bone_hierarchy[bone_idx]
                        print(f"  Local position: {bone_info['position']}")
                        print(f"  Parent index: {bone_info['parent_index']}")
                        
                        # Get parent name if exists
                        if bone_info['parent_index'] != -1:
                            parent_idx = bone_info['parent_index']
                            if parent_idx in extractor.bone_hierarchy:
                                parent_info = extractor.bone_hierarchy[parent_idx]
                                print(f"  Parent bone: '{parent_info['name']}'")
    
    # Also show some raw bone data from the PMX
    print(f"\nRaw bone data from PMX (first 5 bones):")
    for i, bone in enumerate(extractor.pmx_model.bones[:5]):
        print(f"  Bone {i}:")
        print(f"    Name: '{bone.name}' (repr: {repr(bone.name)})")
        if hasattr(bone, 'english_name'):
            print(f"    English name: '{bone.english_name}' (repr: {repr(bone.english_name)})")
        print(f"    Position: {bone.position}")
        print(f"    Parent index: {bone.parent_index}")


if __name__ == "__main__":
    debug_bone_names()