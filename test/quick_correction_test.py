#!/usr/bin/env python
# coding: utf-8
"""
Quick test of corrected bone transformation implementations
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pymeshio.pmx.reader as pmx_reader
import pymeshio.vmd.reader as vmd_reader

# Import all implementations
from bone_animation import get_bone_world_position
from bone_animation_pytransform3d import get_bone_world_position_pt3d
from bone_animation_corrected import (
    get_bone_world_position_corrected,
    get_bone_world_position_corrected_v2,
    get_bone_world_position_corrected_v3
)

def quick_test():
    """Quick test of all implementations on right toe bone."""
    
    # Load test files
    pmx_file = "pdtt.pmx"
    vmd_file = "dan_alivef_01.imo.vmd"
    
    print("🧪 Quick Bone Transformation Correction Test")
    print("=" * 60)
    
    try:
        with open(pmx_file, 'rb') as f:
            pmx_model = pmx_reader.read(f)
        
        with open(vmd_file, 'rb') as f:
            vmd_motion = vmd_reader.read(f)
            
        print("✅ Files loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return
    
    # Test bone and frame
    test_bone = '右つま先'  # Right toe - this was underground in pytransform3d
    test_frame = 3000
    
    implementations = {
        'Original': get_bone_world_position,
        'Pytransform3d': get_bone_world_position_pt3d,
        'Corrected_v1': get_bone_world_position_corrected,
        'Corrected_v2': get_bone_world_position_corrected_v2,
        'Corrected_v3': get_bone_world_position_corrected_v3
    }
    
    print(f"\nTesting bone '{test_bone}' at frame {test_frame}:")
    print("=" * 60)
    
    for impl_name, impl_func in implementations.items():
        try:
            pos = impl_func(pmx_model, vmd_motion, test_frame, test_bone)
            
            # Check if reasonable (not underground)
            reasonable = pos[1] >= -5.0
            status = "✅" if reasonable else f"❌ (Y={pos[1]:.3f} underground)"
            
            print(f"{impl_name:15}: ({pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}) {status}")
            
        except Exception as e:
            print(f"{impl_name:15}: ❌ Error: {e}")
    
    # Expected result from debug output:
    # Original should be around: (-19.825, 2.964, -19.963) ✅ Good
    # Pytransform3d was: (19.791, -12.559, -15.841) ❌ Underground
    # Corrected versions should hopefully fix the underground issue
    
    print("\n🎯 Expected Results:")
    print("- Original: Y ≈ 2.964 (good, near ground)")
    print("- Pytransform3d: Y ≈ -12.559 (bad, underground)")
    print("- Corrected versions: Should have reasonable Y values")
    

if __name__ == "__main__":
    quick_test()