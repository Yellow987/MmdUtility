#!/usr/bin/env python3
"""
Test script for frame 3000 quaternion application functionality.
Demonstrates extracting quaternions and positions from VMD frame 3000,
applying them to the PMX skeleton, and calculating world positions.
"""

import os
import sys
import numpy as np

def test_frame_3000_transforms():
    """Test applying VMD transforms from frame 3000 to skeleton."""
    print("Testing Frame 3000 Quaternion Application...")
    
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
    
    print(f"Using PMX file: {pmx_path}")
    print(f"Using VMD file: {vmd_path}")
    
    try:
        import bone_position_extractor
        extractor = bone_position_extractor.BonePositionExtractor()
        
        # Load files
        print("\nLoading PMX model...")
        extractor.load_pmx(pmx_path)
        
        print("Loading VMD motion...")
        extractor.load_vmd(vmd_path)
        
        # Get motion info to see frame range
        motion_info = extractor.get_motion_info()
        max_frame = motion_info.get('max_frame', 0)
        print(f"VMD max frame: {max_frame}")
        
        if max_frame < 3000:
            print(f"⚠ Warning: VMD only has {max_frame} frames, but testing with frame 3000 anyway")
        
        # Test getting frame transforms for frame 3000
        print("\nGetting frame transforms for frame 3000...")
        frame_transforms = extractor.get_frame_transforms(3000)
        print(f"Found {len(frame_transforms)} bone transforms in frame 3000")
        
        # Show some sample transforms
        if frame_transforms:
            print("Sample bone transforms from frame 3000:")
            sample_bones = list(frame_transforms.keys())[:5]
            for bone_name in sample_bones:
                pos_offset, quaternion = frame_transforms[bone_name]
                print(f"  {bone_name}:")
                print(f"    Position offset: ({pos_offset[0]:.3f}, {pos_offset[1]:.3f}, {pos_offset[2]:.3f})")
                print(f"    Quaternion: ({quaternion.x:.3f}, {quaternion.y:.3f}, {quaternion.z:.3f}, {quaternion.w:.3f})")
        else:
            print("No transforms found for frame 3000")
        
        # Apply transforms to skeleton and get world positions
        print("\nApplying frame 3000 transforms to skeleton...")
        world_positions = extractor.apply_frame_transforms_to_skeleton(3000)
        print(f"Calculated world positions for {len(world_positions)} bones")
        
        # Compare with rest pose positions
        print("\nGetting rest pose positions for comparison...")
        rest_positions = extractor.get_rest_pose_positions()
        
        # Show comparison for some key bones
        print("\nWorld positions after applying frame 3000 transforms:")
        print("Bone Name                | Rest Pose Position      | Frame 3000 Position     | Difference")
        print("-" * 90)
        
        # Sort bones for consistent output
        sorted_bones = sorted(world_positions.keys())[:15]  # Show first 15 bones
        
        for bone_name in sorted_bones:
            world_pos = world_positions[bone_name]
            
            # Find corresponding rest pose (handle unique naming)
            rest_pos = None
            if bone_name in rest_positions:
                rest_pos = rest_positions[bone_name]
            else:
                # Try to find base name without index suffix
                base_name = bone_name.split('_')[0] if '_' in bone_name else bone_name
                if base_name in rest_positions:
                    rest_pos = rest_positions[base_name]
            
            if rest_pos:
                diff = (
                    world_pos[0] - rest_pos[0],
                    world_pos[1] - rest_pos[1], 
                    world_pos[2] - rest_pos[2]
                )
                diff_magnitude = np.sqrt(sum(d*d for d in diff))
                
                print(f"{bone_name:<20} | ({rest_pos[0]:>6.2f},{rest_pos[1]:>6.2f},{rest_pos[2]:>6.2f}) | "
                      f"({world_pos[0]:>6.2f},{world_pos[1]:>6.2f},{world_pos[2]:>6.2f}) | {diff_magnitude:>6.3f}")
            else:
                print(f"{bone_name:<20} | {'N/A':<20} | "
                      f"({world_pos[0]:>6.2f},{world_pos[1]:>6.2f},{world_pos[2]:>6.2f}) | N/A")
        
        print(f"\n✓ Successfully applied frame 3000 transforms to skeleton!")
        print(f"✓ World positions calculated for all {len(world_positions)} bones")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during frame 3000 transform test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run frame 3000 transform test."""
    print("=" * 80)
    print("Frame 3000 Quaternion Application Test")
    print("=" * 80)
    
    success = test_frame_3000_transforms()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Frame 3000 transform test completed successfully!")
        print("The skeleton now has world positions with frame 3000 transforms applied.")
    else:
        print("❌ Frame 3000 transform test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)