#!/usr/bin/env python3
"""
List all bones in a PMX file with their names and positions.
"""

import os
from bone_position_extractor import BonePositionExtractor


def list_all_bones():
    """List all bones in the PMX file."""
    pmx_path = "test/pdtt.pmx"
    if not os.path.exists(pmx_path):
        print(f"Test file not found: {pmx_path}")
        return
    
    print("Loading PMX file and listing all bones...")
    
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    
    print(f"\nTotal bones: {len(extractor.pmx_model.bones)}")
    print("=" * 80)
    
    for i, bone in enumerate(extractor.pmx_model.bones):
        bone_name = bone.name
        if isinstance(bone_name, bytes):
            bone_name = bone_name.decode('utf-8', errors='ignore')
        
        english_name = ""
        if hasattr(bone, 'english_name') and bone.english_name:
            english_name = bone.english_name
            if isinstance(english_name, bytes):
                english_name = english_name.decode('utf-8', errors='ignore')
        
        print(f"Bone {i:3d}: '{bone_name}'")
        if english_name:
            print(f"         English: '{english_name}'")
        print(f"         Position: ({bone.position[0]:.6f}, {bone.position[1]:.6f}, {bone.position[2]:.6f})")
        print(f"         Parent: {bone.parent_index}")
        
        # Check if this bone name contains the problematic ID
        if "#3997a7ac" in bone_name:
            print(f"         *** THIS IS THE PROBLEMATIC BONE! ***")
        
        print()


if __name__ == "__main__":
    list_all_bones()