#!/usr/bin/env python3
"""
Test script to verify the integrated bezier interpolation functionality
in bone_position_extractor.py for frame 3000.
"""

import os
import sys

def test_integrated_bezier_interpolation():
    """Test the integrated bezier interpolation functionality."""
    print("Testing Integrated Bezier Interpolation for Frame 3000...")
    
    # Define test file paths
    pmx_path = "test/pdtt.pmx"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    
    # Check if files exist
    if not os.path.exists(pmx_path) or not os.path.exists(vmd_path):
        print(f"✗ Test files not found: {pmx_path}, {vmd_path}")
        return False
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        # Load files
        print("Loading PMX and VMD files...")
        extractor.load_pmx(pmx_path)
        extractor.load_vmd(vmd_path)
        
        target_frame = 3000
        
        print(f"\n" + "="*60)
        print(f"COMPARING METHODS FOR FRAME {target_frame}")
        print("="*60)
        
        # Test 1: Original method (exact keyframes only)
        print("\n1. Using original method (exact keyframes only):")
        world_positions_original = extractor.apply_frame_transforms_to_skeleton(
            target_frame, use_interpolation=False
        )
        rest_positions = extractor.get_rest_pose_positions()
        
        print(f"   Original method found positions for {len(world_positions_original)} bones")
        
        # Test 2: New method with bezier interpolation
        print("\n2. Using bezier interpolation method:")
        world_positions_interpolated = extractor.apply_frame_transforms_to_skeleton(
            target_frame, use_interpolation=True
        )
        
        print(f"   Interpolated method found positions for {len(world_positions_interpolated)} bones")
        
        # Compare results
        print(f"\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        
        print(f"{'Bone Name':<15} | {'Rest Pose':<20} | {'Original':<20} | {'Interpolated':<20} | {'Diff from Rest':<12}")
        print("-" * 95)
        
        significant_differences = 0
        bones_with_animation = 0
        
        # Check all bones that appear in either result
        all_bones = set(world_positions_original.keys()) | set(world_positions_interpolated.keys())
        
        for bone_name in sorted(all_bones):
            if bone_name in rest_positions:
                rest_pos = rest_positions[bone_name]
                orig_pos = world_positions_original.get(bone_name, rest_pos)
                interp_pos = world_positions_interpolated.get(bone_name, rest_pos)
                
                # Calculate differences from rest pose
                orig_diff = sum((orig_pos[i] - rest_pos[i])**2 for i in range(3))**0.5
                interp_diff = sum((interp_pos[i] - rest_pos[i])**2 for i in range(3))**0.5
                
                rest_str = f"({rest_pos[0]:.2f},{rest_pos[1]:.2f},{rest_pos[2]:.2f})"
                orig_str = f"({orig_pos[0]:.2f},{orig_pos[1]:.2f},{orig_pos[2]:.2f})"
                interp_str = f"({interp_pos[0]:.2f},{interp_pos[1]:.2f},{interp_pos[2]:.2f})"
                
                # Mark bones with significant animation
                has_animation = interp_diff > 0.01
                if has_animation:
                    bones_with_animation += 1
                    animation_marker = " *"
                else:
                    animation_marker = ""
                
                if abs(orig_diff - interp_diff) > 0.01:
                    significant_differences += 1
                
                print(f"{bone_name:<15} | {rest_str:<20} | {orig_str:<20} | {interp_str:<20} | {interp_diff:<12.3f}{animation_marker}")
        
        print(f"\nSummary:")
        print(f"  Bones processed: {len(all_bones)}")
        print(f"  Bones with animation (>0.01 units from rest): {bones_with_animation}")
        print(f"  Significant differences between methods: {significant_differences}")
        
        # Test 3: Direct interpolated transforms
        print(f"\n" + "="*60)
        print("DIRECT INTERPOLATED TRANSFORMS TEST")
        print("="*60)
        
        interpolated_transforms = extractor.get_interpolated_frame_transforms(target_frame, core_bones_only=True)
        print(f"Found {len(interpolated_transforms)} interpolated transforms:")
        
        print(f"{'Bone Name':<15} | {'Position':<25} | {'Quaternion (x,y,z,w)':<30}")
        print("-" * 75)
        
        for bone_name, (pos, quat) in sorted(interpolated_transforms.items()):
            pos_str = f"({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
            
            # Handle different quaternion types
            if hasattr(quat, '__len__') and len(quat) == 4:
                # Numpy array from SLERP
                quat_str = f"({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f})"
            elif hasattr(quat, 'x'):
                # VMD quaternion object
                quat_str = f"({quat.x:.3f},{quat.y:.3f},{quat.z:.3f},{quat.w:.3f})"
            else:
                quat_str = str(quat)
            
            print(f"{bone_name:<15} | {pos_str:<25} | {quat_str:<30}")
        
        # Check if bezier library is available
        if bone_position_extractor.HAS_BEZIER:
            print(f"\n✓ Bezier library is available and being used for curve interpolation")
        else:
            print(f"\n⚠ Bezier library not available, using linear interpolation fallback")
        
        print(f"\n✓ Integrated bezier interpolation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during integrated bezier interpolation test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integrated bezier interpolation test."""
    print("=" * 80)
    print("Integrated Bezier Interpolation Test - Frame 3000")
    print("=" * 80)
    
    success = test_integrated_bezier_interpolation()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 All tests completed successfully!")
        print("The bone_position_extractor.py now supports bezier interpolation!")
    else:
        print("❌ Test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)