#!/usr/bin/env python
# coding: utf-8
"""
Test script to compare numpy vs pytransform3d bone animation implementations
"""

import os
import sys
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pymeshio.pmx.reader as pmx_reader
import pymeshio.vmd.reader as vmd_reader
from bone_animation import get_bone_world_position
from bone_animation_pytransform3d import (
    get_bone_world_position_pt3d,
    compare_implementations,
    get_bone_animation_data_slerp,
    get_bone_animation_data
)

def test_implementations():
    """Test both implementations with sample data"""
    
    # Load test files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    pmx_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.pmx')]
    vmd_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.vmd')]
    
    if not pmx_files or not vmd_files:
        print("❌ Need PMX and VMD files in test directory")
        print(f"Test directory: {test_dir}")
        print(f"PMX files found: {pmx_files}")
        print(f"VMD files found: {vmd_files}")
        return
    
    pmx_path = os.path.join(test_dir, pmx_files[0])
    vmd_path = os.path.join(test_dir, vmd_files[0])
    
    print(f"🔄 Loading {pmx_files[0]} and {vmd_files[0]}")
    
    try:
        pmx_model = pmx_reader.read_from_file(pmx_path)
        vmd_motion = vmd_reader.read_from_file(vmd_path)
        print(f"✅ Files loaded successfully")
        print(f"   Model bones: {len(pmx_model.bones)}")
        print(f"   VMD frames: {len(vmd_motion.motions)}")
    except Exception as e:
        print(f"❌ Failed to load files: {e}")
        return
    
    # Test bones and frames
    test_bones = ['左足首', '右足首', '左つま先', '右つま先']  # ankle and toe bones
    test_frames = [0, 1000, 3000, 5000]
    
    print(f"\n" + "="*100)
    print(f"PYTRANSFORM3D VS NUMPY COMPARISON TEST")
    print(f"="*100)
    
    overall_results = {
        'total_tests': 0,
        'successful_tests': 0,
        'failed_tests': 0,
        'max_difference': 0.0,
        'differences': []
    }
    
    for frame in test_frames:
        print(f"\n{'='*20} FRAME {frame} {'='*20}")
        
        for bone_name in test_bones:
            print(f"\n--- Testing {bone_name} ---")
            
            # Compare implementations
            comparison = compare_implementations(pmx_model, vmd_motion, frame, bone_name)
            overall_results['total_tests'] += 1
            
            # Display results
            if comparison['numpy_error']:
                print(f"❌ NumPy failed: {comparison['numpy_error']}")
                overall_results['failed_tests'] += 1
            else:
                numpy_pos = comparison['numpy_result']
                print(f"NumPy:        ({numpy_pos[0]:10.6f}, {numpy_pos[1]:10.6f}, {numpy_pos[2]:10.6f})")
            
            if comparison['pytransform3d_error']:
                print(f"❌ Pytransform3d failed: {comparison['pytransform3d_error']}")
                overall_results['failed_tests'] += 1
            else:
                pt3d_pos = comparison['pytransform3d_result']
                print(f"Pytransform3d: ({pt3d_pos[0]:10.6f}, {pt3d_pos[1]:10.6f}, {pt3d_pos[2]:10.6f})")
            
            # Show comparison if both succeeded
            if comparison['difference'] is not None:
                overall_results['successful_tests'] += 1
                diff = comparison['difference']
                distance = comparison['distance_difference']
                overall_results['max_difference'] = max(overall_results['max_difference'], distance)
                overall_results['differences'].append(distance)
                
                print(f"Difference:   ({diff[0]:10.6f}, {diff[1]:10.6f}, {diff[2]:10.6f})")
                print(f"Distance:     {distance:.10f}")
                
                if comparison['tolerance_check']:
                    print("✅ Match within tolerance")
                else:
                    print("⚠️  Beyond tolerance")
                    
                # Calculate relative error
                if comparison['numpy_result'] is not None:
                    numpy_magnitude = np.linalg.norm(comparison['numpy_result'])
                    if numpy_magnitude > 1e-10:
                        relative_error = distance / numpy_magnitude
                        print(f"Rel. error:   {relative_error:.10f} ({relative_error*100:.8f}%)")
    
    # Overall summary
    print(f"\n" + "="*100)
    print(f"OVERALL SUMMARY")
    print(f"="*100)
    print(f"Total tests:           {overall_results['total_tests']}")
    print(f"Successful comparisons: {overall_results['successful_tests']}")
    print(f"Failed tests:          {overall_results['failed_tests']}")
    print(f"Success rate:          {overall_results['successful_tests']/overall_results['total_tests']*100:.1f}%")
    
    if overall_results['differences']:
        differences = np.array(overall_results['differences'])
        print(f"Maximum difference:    {overall_results['max_difference']:.10f}")
        print(f"Mean difference:       {np.mean(differences):.10f}")
        print(f"Std difference:        {np.std(differences):.10f}")
        
        tolerance_passed = sum(1 for d in differences if d < 1e-6)
        print(f"Within tolerance:      {tolerance_passed}/{len(differences)} ({tolerance_passed/len(differences)*100:.1f}%)")
        
        if overall_results['max_difference'] < 1e-6:
            print("🎉 ALL RESULTS MATCH WITHIN TOLERANCE!")
        elif overall_results['max_difference'] < 1e-3:
            print("⚠️  Small differences detected - generally acceptable")
        else:
            print("❌ Significant differences - requires investigation")
    
    return overall_results


def test_interpolation_comparison():
    """Test linear vs SLERP interpolation methods"""
    
    print(f"\n" + "="*100)
    print(f"INTERPOLATION METHOD COMPARISON (Linear vs SLERP)")
    print(f"="*100)
    
    # Load test files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pmx_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.pmx')]
    vmd_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.vmd')]
    
    if not pmx_files or not vmd_files:
        print("❌ Need PMX and VMD files in test directory")
        return
    
    pmx_path = os.path.join(test_dir, pmx_files[0])
    vmd_path = os.path.join(test_dir, vmd_files[0])
    
    try:
        pmx_model = pmx_reader.read_from_file(pmx_path)
        vmd_motion = vmd_reader.read_from_file(vmd_path)
    except Exception as e:
        print(f"❌ Failed to load files: {e}")
        return
    
    test_bones = ['左足首', '右足首']
    test_frames = [1500, 2500, 3500]  # Frames likely between keyframes
    
    for frame in test_frames:
        print(f"\n{'='*20} FRAME {frame} {'='*20}")
        
        for bone_name in test_bones:
            print(f"\n--- {bone_name} ---")
            
            try:
                # Linear interpolation (original)
                from bone_animation import get_bone_animation_data
                anim_pos_linear, anim_quat_linear = get_bone_animation_data(vmd_motion, bone_name, frame)
                
                # SLERP interpolation (pytransform3d)
                anim_pos_slerp, anim_quat_slerp = get_bone_animation_data_slerp(vmd_motion, bone_name, frame)
                
                print(f"Linear Position:   ({anim_pos_linear.x:.6f}, {anim_pos_linear.y:.6f}, {anim_pos_linear.z:.6f})")
                print(f"SLERP Position:    ({anim_pos_slerp.x:.6f}, {anim_pos_slerp.y:.6f}, {anim_pos_slerp.z:.6f})")
                
                print(f"Linear Quaternion: ({anim_quat_linear.x:.6f}, {anim_quat_linear.y:.6f}, {anim_quat_linear.z:.6f}, {anim_quat_linear.w:.6f})")
                print(f"SLERP Quaternion:  ({anim_quat_slerp.x:.6f}, {anim_quat_slerp.y:.6f}, {anim_quat_slerp.z:.6f}, {anim_quat_slerp.w:.6f})")
                
                # Calculate differences
                pos_diff = np.array([
                    anim_pos_linear.x - anim_pos_slerp.x,
                    anim_pos_linear.y - anim_pos_slerp.y,
                    anim_pos_linear.z - anim_pos_slerp.z
                ])
                pos_distance = np.linalg.norm(pos_diff)
                
                quat_diff = np.array([
                    anim_quat_linear.x - anim_quat_slerp.x,
                    anim_quat_linear.y - anim_quat_slerp.y,
                    anim_quat_linear.z - anim_quat_slerp.z,
                    anim_quat_linear.w - anim_quat_slerp.w
                ])
                quat_distance = np.linalg.norm(quat_diff)
                
                print(f"Position difference:  {pos_distance:.10f}")
                print(f"Quaternion difference: {quat_distance:.10f}")
                
                if pos_distance < 1e-6 and quat_distance < 1e-6:
                    print("✅ Interpolation methods produce identical results")
                else:
                    print("📊 Interpolation methods differ - SLERP may be more accurate for rotations")
                    
            except Exception as e:
                print(f"❌ Interpolation test failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Pytransform3d vs NumPy Comparison Test")
    
    try:
        # Test main implementations
        results = test_implementations()
        
        # Test interpolation methods
        test_interpolation_comparison()
        
        print(f"\n🎯 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()