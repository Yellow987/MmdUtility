#!/usr/bin/env python3
"""
Test script to examine quaternion animation data specifically around frame 3000.
Shows which bones have keyframes at frame 3000 and nearby frames.
"""

import os
import sys
import numpy as np

def analyze_frame_3000_animation():
    """Analyze VMD animation data around frame 3000."""
    print("Analyzing VMD Quaternion Animation Data for Frame 3000...")
    
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
        
        # Get motion info
        motion_info = extractor.get_motion_info()
        max_frame = motion_info.get('max_frame', 0)
        print(f"VMD max frame: {max_frame}")
        print(f"Total bone motions: {motion_info.get('bone_motions', 0)}")
        
        # Define core bones to filter for (same as in bone_position_extractor)
        core_bone_names = {
            "", "センター", "グルーブ", "腰", "上半身", "上半身2",
            "首", "頭",
            "左肩", "右肩", "左腕", "右腕", "左ひじ", "右ひじ", "左手首", "右手首",
            "左足", "右足", "左ひざ", "右ひざ", "左足首", "右足首", "左つま先", "右つま先",
        }
        
        # Direct analysis of VMD data
        print("\n" + "="*80)
        print("DIRECT VMD DATA ANALYSIS FOR FRAME 3000 (CORE BONES ONLY)")
        print("="*80)
        
        # Check for exact frame 3000 keyframes (core bones only)
        frame_3000_motions = []
        for motion in extractor.vmd_motion.motions:
            if motion.frame == 3000:
                bone_name = motion.name
                if isinstance(bone_name, bytes):
                    bone_name = bone_name.decode('utf-8', errors='ignore')
                
                # Only include core bones
                if bone_name in core_bone_names:
                    frame_3000_motions.append(motion)
        
        print(f"\nBones with EXACT keyframes at frame 3000: {len(frame_3000_motions)}")
        
        if frame_3000_motions:
            print("\nFrame 3000 Keyframe Data:")
            print("-" * 80)
            print(f"{'Bone Name':<20} | {'Position (X,Y,Z)':<25} | {'Quaternion (X,Y,Z,W)':<30}")
            print("-" * 80)
            
            for motion in frame_3000_motions:
                bone_name = motion.name
                if isinstance(bone_name, bytes):
                    bone_name = bone_name.decode('utf-8', errors='ignore')
                
                pos = f"({motion.pos.x:.3f},{motion.pos.y:.3f},{motion.pos.z:.3f})"
                quat = f"({motion.q.x:.3f},{motion.q.y:.3f},{motion.q.z:.3f},{motion.q.w:.3f})"
                
                print(f"{bone_name:<20} | {pos:<25} | {quat:<30}")
        
        # Check for keyframes in a range around frame 3000 (core bones only)
        print(f"\n" + "="*60)
        print("KEYFRAMES IN RANGE 2990-3010 (CORE BONES ONLY)")
        print("="*60)
        
        nearby_frames = {}
        for motion in extractor.vmd_motion.motions:
            if 2990 <= motion.frame <= 3010:
                bone_name = motion.name
                if isinstance(bone_name, bytes):
                    bone_name = bone_name.decode('utf-8', errors='ignore')
                
                # Only include core bones
                if bone_name in core_bone_names:
                    frame_num = motion.frame
                    if frame_num not in nearby_frames:
                        nearby_frames[frame_num] = []
                        
                    nearby_frames[frame_num].append({
                        'bone': bone_name,
                        'pos': (motion.pos.x, motion.pos.y, motion.pos.z),
                        'quat': (motion.q.x, motion.q.y, motion.q.z, motion.q.w)
                    })
        
        if nearby_frames:
            print(f"Found keyframes in {len(nearby_frames)} frames between 2990-3010:")
            
            for frame_num in sorted(nearby_frames.keys()):
                motions = nearby_frames[frame_num]
                print(f"\nFrame {frame_num}: {len(motions)} bones")
                
                # Show first few bones for each frame
                for i, motion_data in enumerate(motions[:5]):  # Show first 5 bones
                    bone = motion_data['bone']
                    pos = motion_data['pos']
                    quat = motion_data['quat']
                    print(f"  {bone:<15}: pos({pos[0]:>6.3f},{pos[1]:>6.3f},{pos[2]:>6.3f}) "
                          f"quat({quat[0]:>6.3f},{quat[1]:>6.3f},{quat[2]:>6.3f},{quat[3]:>6.3f})")
                
                if len(motions) > 5:
                    print(f"  ... and {len(motions)-5} more bones")
        else:
            print("No keyframes found in range 2990-3010")
        
        # Show keyframe distribution statistics (core bones only)
        print(f"\n" + "="*60)
        print("KEYFRAME DISTRIBUTION ANALYSIS (CORE BONES ONLY)")
        print("="*60)
        
        # Collect all frames for analysis (core bones only)
        all_frames = set()
        bone_frame_counts = {}
        
        for motion in extractor.vmd_motion.motions:
            bone_name = motion.name
            if isinstance(bone_name, bytes):
                bone_name = bone_name.decode('utf-8', errors='ignore')
            
            # Only analyze core bones
            if bone_name in core_bone_names:
                frame_num = motion.frame
                all_frames.add(frame_num)
                
                if bone_name not in bone_frame_counts:
                    bone_frame_counts[bone_name] = 0
                bone_frame_counts[bone_name] += 1
        
        if all_frames:
            print(f"Total unique frames with keyframes (core bones): {len(all_frames)}")
            print(f"Frame range: {min(all_frames)} to {max(all_frames)}")
            
            # Show bones with most keyframes
            print(f"\nCore bones by keyframe count:")
            sorted_bones = sorted(bone_frame_counts.items(), key=lambda x: x[1], reverse=True)
            for bone_name, count in sorted_bones:
                print(f"  {bone_name:<20}: {count:>5} keyframes")
        else:
            print("No keyframes found for core bones")
        
        # Find closest keyframes to frame 3000
        print(f"\nClosest keyframes to frame 3000:")
        frames_list = sorted(list(all_frames))
        
        # Find frames before and after 3000
        before_3000 = [f for f in frames_list if f < 3000]
        after_3000 = [f for f in frames_list if f > 3000]
        
        if before_3000:
            closest_before = max(before_3000)
            print(f"  Closest frame before 3000: {closest_before} (gap: {3000 - closest_before} frames)")
        
        if after_3000:
            closest_after = min(after_3000)
            print(f"  Closest frame after 3000: {closest_after} (gap: {closest_after - 3000} frames)")
        
        if not before_3000 and not after_3000:
            print("  No keyframes found for core bones around frame 3000")
        
        # Test the get_frame_transforms method
        print(f"\n" + "="*60)
        print("TESTING get_frame_transforms() METHOD")
        print("="*60)
        
        frame_transforms = extractor.get_frame_transforms(3000)
        print(f"get_frame_transforms(3000) returned {len(frame_transforms)} transforms")
        
        if frame_transforms:
            print("Frame 3000 transforms from get_frame_transforms():")
            for bone_name, (pos_offset, quaternion) in frame_transforms.items():
                print(f"  {bone_name:<15}: pos{pos_offset} quat({quaternion.x:.3f},{quaternion.y:.3f},{quaternion.z:.3f},{quaternion.w:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during quaternion animation analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quaternion animation analysis."""
    print("=" * 80)
    print("VMD Quaternion Animation Analysis - Frame 3000")
    print("=" * 80)
    
    success = analyze_frame_3000_animation()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 Quaternion animation analysis completed!")
    else:
        print("❌ Quaternion animation analysis failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)