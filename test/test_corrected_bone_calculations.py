#!/usr/bin/env python
# coding: utf-8
"""
Test script to verify corrected bone position calculations
"""

import os
import sys
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pymeshio.pmx.reader as pmx_reader
import pymeshio.vmd.reader as vmd_reader

# Import both implementations for comparison
from bone_animation_pytransform3d import get_bone_world_position_pt3d
from bone_animation import get_bone_world_position


def test_corrected_foot_positions():
    """Test that corrected bone calculations produce reasonable foot positions"""
    
    print("=== Testing Corrected Bone Position Calculations ===\n")
    
    # Load test files
    pmx_file = "pdtt.pmx"
    vmd_file = "dan_alivef_01.imo.vmd"
    
    if not os.path.exists(pmx_file):
        print(f"ERROR: PMX file '{pmx_file}' not found!")
        return False
    
    if not os.path.exists(vmd_file):
        print(f"ERROR: VMD file '{vmd_file}' not found!")
        return False
    
    # Load models
    print(f"Loading PMX: {pmx_file}")
    with open(pmx_file, 'rb') as f:
        pmx_model = pmx_reader.read(f)
    
    print(f"Loading VMD: {vmd_file}")
    with open(vmd_file, 'rb') as f:
        vmd_motion = vmd_reader.read(f)
    
    # Test frame
    test_frame = 3000
    
    # Test bones (feet)
    foot_bones = [
        '左足首',  # Left ankle
        '右足首',  # Right ankle  
        '左つま先', # Left toe
        '右つま先'  # Right toe
    ]
    
    print(f"\nTesting foot positions at frame {test_frame}:")
    print("=" * 60)
    
    success = True
    results = {}
    
    for bone_name in foot_bones:
        try:
            # Calculate with corrected pytransform3d method
            pos_corrected = get_bone_world_position_pt3d(pmx_model, vmd_motion, test_frame, bone_name)
            
            results[bone_name] = {
                'corrected': pos_corrected
            }
            
            print(f"{bone_name:12}: ({pos_corrected[0]:8.3f}, {pos_corrected[1]:8.3f}, {pos_corrected[2]:8.3f})")
            
            # Basic validation: feet should be near ground level (Y > -5 and Y < 20)
            if pos_corrected[1] < -5.0:
                print(f"  ❌ WARNING: {bone_name} Y position {pos_corrected[1]:.3f} is too low (below ground)")
                success = False
            elif pos_corrected[1] > 20.0:
                print(f"  ❌ WARNING: {bone_name} Y position {pos_corrected[1]:.3f} is too high")
                success = False
            else:
                print(f"  ✅ Y position {pos_corrected[1]:.3f} looks reasonable")
                
        except Exception as e:
            print(f"  ❌ ERROR calculating {bone_name}: {e}")
            success = False
    
    # Check if left/right feet have different positions (for dancing motion)
    print("\n" + "=" * 60)
    print("Checking foot position differences:")
    
    if '左足首' in results and '右足首' in results:
        left_ankle = results['左足首']['corrected']
        right_ankle = results['右足首']['corrected']
        
        y_diff = abs(left_ankle[1] - right_ankle[1])
        print(f"Left/Right ankle Y difference: {y_diff:.3f}")
        
        if y_diff < 0.1:
            print("  ❌ WARNING: Left and right ankles have nearly identical Y positions")
            print("             This suggests the transformation calculations may still be incorrect")
            success = False
        else:
            print("  ✅ Left and right ankles have different Y positions - good!")
    
    if '左つま先' in results and '右つま先' in results:
        left_toe = results['左つま先']['corrected']
        right_toe = results['右つま先']['corrected']
        
        y_diff = abs(left_toe[1] - right_toe[1])
        print(f"Left/Right toe Y difference: {y_diff:.3f}")
        
        if y_diff < 0.1:
            print("  ❌ WARNING: Left and right toes have nearly identical Y positions")
            success = False
        else:
            print("  ✅ Left and right toes have different Y positions - good!")
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ SUCCESS: All bone position calculations look reasonable!")
        print("✅ The transformation fixes appear to be working correctly.")
    else:
        print("❌ ISSUES DETECTED: Some bone positions still look incorrect.")
        print("❌ Additional debugging may be needed.")
    
    return success


def compare_before_after_implementations():
    """Compare old vs new implementation results"""
    
    print("\n=== Comparing Old vs New Implementations ===\n")
    
    # Load test files
    pmx_file = "pdtt.pmx"
    vmd_file = "dan_alivef_01.imo.vmd"
    
    with open(pmx_file, 'rb') as f:
        pmx_model = pmx_reader.read(f)
    with open(vmd_file, 'rb') as f:
        vmd_motion = vmd_reader.read(f)
    
    test_frame = 3000
    test_bone = '左足首'  # Left ankle
    
    try:
        # Get results from both implementations
        pos_old = get_bone_world_position(pmx_model, vmd_motion, test_frame, test_bone)
        pos_new = get_bone_world_position_pt3d(pmx_model, vmd_motion, test_frame, test_bone)
        
        print(f"Bone: {test_bone}, Frame: {test_frame}")
        print(f"Old implementation: ({pos_old[0]:8.3f}, {pos_old[1]:8.3f}, {pos_old[2]:8.3f})")
        print(f"New implementation: ({pos_new[0]:8.3f}, {pos_new[1]:8.3f}, {pos_new[2]:8.3f})")
        
        # Calculate differences
        diff = (pos_new[0] - pos_old[0], pos_new[1] - pos_old[1], pos_new[2] - pos_old[2])
        distance = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        
        print(f"Difference:         ({diff[0]:8.3f}, {diff[1]:8.3f}, {diff[2]:8.3f})")
        print(f"Distance:           {distance:.3f}")
        
        if distance > 0.1:
            print("✅ Implementations show significant difference - transformation fix is working!")
        else:
            print("❌ Implementations show similar results - fix may not be working")
            
    except Exception as e:
        print(f"Error comparing implementations: {e}")


if __name__ == "__main__":
    # Change to test directory
    os.chdir(os.path.dirname(__file__))
    
    # Run tests
    test_corrected_foot_positions()
    compare_before_after_implementations()