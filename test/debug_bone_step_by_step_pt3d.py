#!/usr/bin/env python
# coding: utf-8
"""
Step-by-step bone position debug script using Pytransform3d
"""

import os
import sys
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pymeshio.pmx.reader as pmx_reader
import pymeshio.vmd.reader as vmd_reader
import pymeshio.pmx as pmx

# Import pytransform3d implementation
from bone_animation_pytransform3d import (
    get_bone_world_position_pt3d,
    get_bone_animation_data_slerp
)

# Import shared components from original
from bone_animation import (
    BoneHierarchyWalker,
    get_bone_animation_data
)

# Pytransform3d imports for detailed analysis
from pytransform3d.transformations import transform_from, concat
from pytransform3d.rotations import matrix_from_quaternion

def debug_bone_detailed_pt3d(pmx_model, vmd_motion, frame, bone_name):
    """Debug bone calculation step by step using pytransform3d"""
    
    print(f"\n=== DEBUGGING BONE '{bone_name}' AT FRAME {frame} (PYTRANSFORM3D) ===")
    
    # Step 1: Find bone
    bone_result = BoneHierarchyWalker.find_bone_by_name(pmx_model.bones, bone_name)
    if bone_result is None:
        print(f"ERROR: Bone '{bone_name}' not found!")
        return
    
    bone_index, target_bone = bone_result
    print(f"Found bone at index {bone_index}: {target_bone.name}")
    print(f"Bone rest position: ({target_bone.position.x:.3f}, {target_bone.position.y:.3f}, {target_bone.position.z:.3f})")
    
    # Step 2: Get bone chain
    bone_chain = BoneHierarchyWalker.get_bone_chain_to_root(pmx_model.bones, bone_index)
    print(f"Bone hierarchy chain: {bone_chain}")
    
    # Step 3: Show each bone in chain with incremental world positions using pytransform3d
    print(f"\nBone chain details (using pytransform3d):")
    
    # Find センター bone position for relative position calculations
    center_bone_result = BoneHierarchyWalker.find_bone_by_name(pmx_model.bones, 'センター')
    center_world_pos = None
    if center_bone_result is not None:
        try:
            center_world_pos = get_bone_world_position_pt3d(pmx_model, vmd_motion, frame, 'センター')
            print(f"センター bone world position: ({center_world_pos[0]:.3f}, {center_world_pos[1]:.3f}, {center_world_pos[2]:.3f})")
        except Exception as e:
            print(f"Warning: Could not get センター bone position: {e}")
            center_world_pos = None
    else:
        print("Warning: センター bone not found in model")
    
    # Start with identity transformation matrix
    world_transform = np.eye(4, dtype=float)
    
    for i, chain_bone_index in enumerate(bone_chain):
        bone = pmx_model.bones[chain_bone_index]
        
        # Get animation data for this bone using SLERP interpolation
        anim_pos, anim_quat = get_bone_animation_data_slerp(vmd_motion, bone.name, frame)
        
        # Calculate local position (relative to parent) - same logic as original
        if bone.parent_index == -1:
            # Root bone - local position equals world position
            local_pos_x, local_pos_y, local_pos_z = bone.position.x, bone.position.y, bone.position.z
        else:
            # Child bone - local position = bone_world_pos - parent_world_pos
            parent_bone = pmx_model.bones[bone.parent_index]
            local_pos_x = bone.position.x - parent_bone.position.x
            local_pos_y = bone.position.y - parent_bone.position.y
            local_pos_z = bone.position.z - parent_bone.position.z
        
        # Create transformations using pytransform3d functions
        
        # 1. Rest translation (local bone position)
        rest_pos = np.array([local_pos_x, local_pos_y, local_pos_z])
        rest_transform = transform_from(R=np.eye(3), p=rest_pos)
        
        # 2. Rotation from animation quaternion
        q_array = np.array([anim_quat.w, anim_quat.x, anim_quat.y, anim_quat.z])
        rotation_matrix = matrix_from_quaternion(q_array)
        rotation_transform = transform_from(R=rotation_matrix, p=np.zeros(3))
        
        # 3. Animation translation
        anim_pos_array = np.array([anim_pos.x, anim_pos.y, anim_pos.z])
        anim_transform = transform_from(R=np.eye(3), p=anim_pos_array)
        
        # Combine transformations: Rest * Rotation * Animation
        bone_transform = concat(rest_transform, rotation_transform, anim_transform)
        
        # Accumulate transformation
        world_transform = concat(world_transform, bone_transform)
        
        # Extract current world position
        current_world_pos = world_transform[:3, 3]
        
        # Calculate relative position to センター bone
        relative_pos_to_center = None
        if center_world_pos is not None:
            relative_pos_to_center = current_world_pos - np.array(center_world_pos)
        
        print(f"  {i}: Index {chain_bone_index} - '{bone.name}' (parent: {bone.parent_index})")
        print(f"      Rest pos (world): ({bone.position.x:.3f}, {bone.position.y:.3f}, {bone.position.z:.3f})")
        print(f"      Local pos: ({local_pos_x:.3f}, {local_pos_y:.3f}, {local_pos_z:.3f})")
        print(f"      Anim pos: ({anim_pos.x:.3f}, {anim_pos.y:.3f}, {anim_pos.z:.3f})")
        print(f"      Anim quat: ({anim_quat.x:.3f}, {anim_quat.y:.3f}, {anim_quat.z:.3f}, {anim_quat.w:.3f})")
        print(f"      World pos (pt3d): ({current_world_pos[0]:.3f}, {current_world_pos[1]:.3f}, {current_world_pos[2]:.3f})")
        
        if relative_pos_to_center is not None:
            print(f"      Relative to センター: ({relative_pos_to_center[0]:.3f}, {relative_pos_to_center[1]:.3f}, {relative_pos_to_center[2]:.3f})")
        else:
            print(f"      Relative to センター: (unavailable)")
        
        # Show individual transformation matrices for debugging
        print(f"      Rest transform:")
        print(f"        [[{rest_transform[0,0]:.3f}, {rest_transform[0,1]:.3f}, {rest_transform[0,2]:.3f}, {rest_transform[0,3]:.3f}]]")
        print(f"        [[{rest_transform[1,0]:.3f}, {rest_transform[1,1]:.3f}, {rest_transform[1,2]:.3f}, {rest_transform[1,3]:.3f}]]")
        print(f"        [[{rest_transform[2,0]:.3f}, {rest_transform[2,1]:.3f}, {rest_transform[2,2]:.3f}, {rest_transform[2,3]:.3f}]]")
        print(f"        [[{rest_transform[3,0]:.3f}, {rest_transform[3,1]:.3f}, {rest_transform[3,2]:.3f}, {rest_transform[3,3]:.3f}]]")
        
        print(f"      Rotation transform (from quaternion):")
        print(f"        [[{rotation_transform[0,0]:.3f}, {rotation_transform[0,1]:.3f}, {rotation_transform[0,2]:.3f}, {rotation_transform[0,3]:.3f}]]")
        print(f"        [[{rotation_transform[1,0]:.3f}, {rotation_transform[1,1]:.3f}, {rotation_transform[1,2]:.3f}, {rotation_transform[1,3]:.3f}]]")
        print(f"        [[{rotation_transform[2,0]:.3f}, {rotation_transform[2,1]:.3f}, {rotation_transform[2,2]:.3f}, {rotation_transform[2,3]:.3f}]]")
        print(f"        [[{rotation_transform[3,0]:.3f}, {rotation_transform[3,1]:.3f}, {rotation_transform[3,2]:.3f}, {rotation_transform[3,3]:.3f}]]")
    
    # Step 4: Get animation data for target bone
    print(f"\n--- Animation Data for '{bone_name}' ---")
    anim_pos, anim_quat = get_bone_animation_data_slerp(vmd_motion, bone_name, frame)
    print(f"Animation position (SLERP): ({anim_pos.x:.3f}, {anim_pos.y:.3f}, {anim_pos.z:.3f})")
    print(f"Animation quaternion (SLERP): ({anim_quat.x:.3f}, {anim_quat.y:.3f}, {anim_quat.z:.3f}, {anim_quat.w:.3f})")
    
    # Compare with linear interpolation
    anim_pos_linear, anim_quat_linear = get_bone_animation_data(vmd_motion, bone_name, frame)
    print(f"Animation position (Linear): ({anim_pos_linear.x:.3f}, {anim_pos_linear.y:.3f}, {anim_pos_linear.z:.3f})")
    print(f"Animation quaternion (Linear): ({anim_quat_linear.x:.3f}, {anim_quat_linear.y:.3f}, {anim_quat_linear.z:.3f}, {anim_quat_linear.w:.3f})")
    
    # Show difference between interpolation methods
    pos_diff = np.array([
        anim_pos.x - anim_pos_linear.x,
        anim_pos.y - anim_pos_linear.y,
        anim_pos.z - anim_pos_linear.z
    ])
    quat_diff = np.array([
        anim_quat.x - anim_quat_linear.x,
        anim_quat.y - anim_quat_linear.y,
        anim_quat.z - anim_quat_linear.z,
        anim_quat.w - anim_quat_linear.w
    ])
    
    print(f"Position difference (SLERP - Linear): ({pos_diff[0]:.6f}, {pos_diff[1]:.6f}, {pos_diff[2]:.6f})")
    print(f"Quaternion difference (SLERP - Linear): ({quat_diff[0]:.6f}, {quat_diff[1]:.6f}, {quat_diff[2]:.6f}, {quat_diff[3]:.6f})")
    
    # Step 5: Check if animation data exists in VMD
    print(f"\n--- VMD Data Analysis for '{bone_name}' ---")
    
    # Properly handle encoding - bone_name is already decoded, VMD frame names are Shift-JIS bytes
    bone_name_shift_jis = None
    if isinstance(bone_name, str):
        try:
            bone_name_shift_jis = bone_name.encode('shift-jis')
        except UnicodeEncodeError:
            try:
                bone_name_shift_jis = bone_name.encode('utf-8')
            except UnicodeEncodeError:
                bone_name_shift_jis = bone_name.encode('utf-8', errors='replace')
    else:
        bone_name_shift_jis = bone_name
    
    bone_frames = []
    for vmd_frame in vmd_motion.motions:
        frame_name = vmd_frame.name
        if frame_name == bone_name_shift_jis:
            bone_frames.append(vmd_frame)
    
    print(f"Total VMD keyframes for '{bone_name}': {len(bone_frames)}")
    if bone_frames:
        bone_frames.sort(key=lambda f: f.frame)
        print(f"Frame range: {bone_frames[0].frame} to {bone_frames[-1].frame}")
        
        # Find frames around our target frame
        nearby_frames = [f for f in bone_frames if abs(f.frame - frame) <= 5]
        print(f"Frames near {frame}: {[f.frame for f in nearby_frames]}")
        
        if nearby_frames:
            for f in nearby_frames:
                print(f"  Frame {f.frame}: pos=({f.pos.x:.3f}, {f.pos.y:.3f}, {f.pos.z:.3f}) rot=({f.q.x:.3f}, {f.q.y:.3f}, {f.q.z:.3f}, {f.q.w:.3f})")
    
    # Step 6: Calculate final world position using pytransform3d
    print(f"\n--- World Position Calculation (Pytransform3d) ---")
    try:
        world_pos = get_bone_world_position_pt3d(pmx_model, vmd_motion, frame, bone_name)
        print(f"Final world position: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})")
    except Exception as e:
        print(f"ERROR calculating world position: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Load test files
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    pmx_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.pmx')]
    vmd_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.vmd')]
    
    if not pmx_files or not vmd_files:
        print("Need PMX and VMD files in test directory")
        return
    
    pmx_path = os.path.join(test_dir, pmx_files[0])
    vmd_path = os.path.join(test_dir, vmd_files[0])
    
    print(f"Loading {pmx_files[0]} and {vmd_files[0]}")
    
    pmx_model = pmx_reader.read_from_file(pmx_path)
    vmd_motion = vmd_reader.read_from_file(vmd_path)
    
    # Show which bones actually have animation data
    print(f"\n=== VMD ANIMATION DATA ANALYSIS ===")
    print(f"Total VMD motion frames: {len(vmd_motion.motions)}")
    
    # Group by bone name
    bone_frame_counts = {}
    bone_name_mapping = {}  # Store both decoded and raw bytes
    
    for vmd_frame in vmd_motion.motions:
        bone_name_raw = vmd_frame.name
        
        # Try multiple encodings to decode Japanese text
        bone_name_decoded = None
        if isinstance(bone_name_raw, bytes):
            # Try common encodings for VMD files
            for encoding in ['shift-jis', 'utf-8', 'cp932']:
                try:
                    bone_name_decoded = bone_name_raw.decode(encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            # Fallback to showing raw bytes if all decodings fail
            if bone_name_decoded is None:
                bone_name_decoded = f"<bytes:{bone_name_raw}>"
        else:
            bone_name_decoded = str(bone_name_raw)
        
        # Store mapping for debugging
        bone_name_mapping[bone_name_decoded] = bone_name_raw
        
        if bone_name_decoded not in bone_frame_counts:
            bone_frame_counts[bone_name_decoded] = []
        bone_frame_counts[bone_name_decoded].append(vmd_frame.frame)
    
    print(f"Bones with animation data: {len(bone_frame_counts)}")
    print("\nAll animated bones:")
    for bone_name, frames in sorted(bone_frame_counts.items(), key=lambda x: len(x[1]), reverse=True):
        frame_range = f"{min(frames)}-{max(frames)}" if len(frames) > 1 else str(frames[0])
        raw_bytes = bone_name_mapping[bone_name]
        if isinstance(raw_bytes, bytes):
            raw_display = f" [bytes: {raw_bytes[:20]}{'...' if len(raw_bytes) > 20 else ''}]"
        else:
            raw_display = ""
        print(f"  {bone_name:30s}: {len(frames):5d} frames ({frame_range}){raw_display}")
    
    # Print all bone initial positions
    print(f"\n=== ALL BONE INITIAL POSITIONS ===")
    print(f"Total bones: {len(pmx_model.bones)}")
    
    for i, bone in enumerate(pmx_model.bones):
        print(f"{i:3d}: {bone.name:25s} pos=({bone.position.x:8.3f}, {bone.position.y:8.3f}, {bone.position.z:8.3f}) parent={bone.parent_index}")
    
    frame = 3000
    
    # Test bones to analyze
    test_bones = ['左足首', '右足首', '左つま先', '右つま先']  # left ankle, right ankle, left toe, right toe
    
    print(f"\n" + "="*100)
    print(f"PYTRANSFORM3D BONE ANALYSIS - FRAME {frame}")
    print(f"="*100)
    
    # Debug each bone with detailed analysis
    for bone_name in test_bones:
        debug_bone_detailed_pt3d(pmx_model, vmd_motion, frame, bone_name)
    
    # Summary comparison
    print(f"\n" + "="*100)
    print(f"SUMMARY - FRAME {frame}")
    print(f"="*100)
    
    positions = {}
    
    for bone_name in test_bones:
        try:
            pos = get_bone_world_position_pt3d(pmx_model, vmd_motion, frame, bone_name)
            positions[bone_name] = pos
        except Exception as e:
            positions[bone_name] = f"ERROR: {e}"
    
    print(f"\nPytransform3d Results:")
    for bone_name in test_bones:
        pos = positions[bone_name]
        if isinstance(pos, tuple):
            print(f"  {bone_name:12s}: ({pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f})")
        else:
            print(f"  {bone_name:12s}: {pos}")
    
    # Validation checks
    if isinstance(positions.get('左足首'), tuple) and isinstance(positions.get('右足首'), tuple):
        left_ankle_pos = positions['左足首']
        right_ankle_pos = positions['右足首']
        ankle_y_diff = abs(left_ankle_pos[1] - right_ankle_pos[1])
        
        print(f"\n" + "-"*60)
        print(f"VALIDATION CHECKS:")
        print(f"  Ankle Y difference: {ankle_y_diff:.6f}")
        
        if ankle_y_diff < 0.001:
            print("  ⚠️  PROBLEM: Identical ankle Y coordinates!")
        else:
            print("  ✅ Ankle Y coordinates are different - good!")
        
        if isinstance(positions.get('左つま先'), tuple) and isinstance(positions.get('右つま先'), tuple):
            left_toe_pos = positions['左つま先']
            right_toe_pos = positions['右つま先']
            toe_y_diff = abs(left_toe_pos[1] - right_toe_pos[1])
            
            print(f"  Toe Y difference: {toe_y_diff:.6f}")
            
            if toe_y_diff < 0.001:
                print("  ⚠️  PROBLEM: Identical toe Y coordinates!")
            else:
                print("  ✅ Toe Y coordinates are different - good!")


if __name__ == "__main__":
    main()