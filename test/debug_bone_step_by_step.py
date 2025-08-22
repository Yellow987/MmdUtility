#!/usr/bin/env python
# coding: utf-8
"""
Step-by-step bone position debug script
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pymeshio.pmx.reader as pmx_reader
import pymeshio.vmd.reader as vmd_reader
import pymeshio.pmx as pmx
import bone_animation
from bone_animation import (
    BoneHierarchyWalker,
    get_bone_animation_data,
    get_bone_world_position
)

def debug_bone_detailed(pmx_model, vmd_motion, frame, bone_name):
    """Debug bone calculation step by step"""
    
    print(f"\n=== DEBUGGING BONE '{bone_name}' AT FRAME {frame} ===")
    
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
    
    # Step 3: Show each bone in chain with incremental world positions
    print("\nBone chain details:")
    import numpy as np
    
    # Start with identity transformation
    world_transform = np.eye(4, dtype=float)
    
    for i, chain_bone_index in enumerate(bone_chain):
        bone = pmx_model.bones[chain_bone_index]
        # Get animation data for this bone
        anim_pos, anim_quat = get_bone_animation_data(vmd_motion, bone.name, frame)
        
        # Calculate this bone's transformation (same logic as in get_bone_world_position)
        rest_pos = bone.position
        
        # Create translation matrix for rest position
        rest_translation = np.eye(4, dtype=float)
        rest_translation[0:3, 3] = [rest_pos.x, rest_pos.y, rest_pos.z]
        
        # Create translation matrix for animation position
        anim_translation = np.eye(4, dtype=float)
        anim_translation[0:3, 3] = [anim_pos.x, anim_pos.y, anim_pos.z]
        
        # Get rotation matrix from quaternion
        rotation_matrix = anim_quat.getMatrix()
        
        # Combine: Translation * Rotation * Rest_Translation
        bone_transform = np.dot(np.dot(anim_translation, rotation_matrix), rest_translation)
        
        # Accumulate transformation
        world_transform = np.dot(world_transform, bone_transform)
        
        # Extract current world position
        current_world_pos = world_transform[0:3, 3]
        
        print(f"  {i}: Index {chain_bone_index} - '{bone.name}' (parent: {bone.parent_index})")
        print(f"      Rest pos: ({bone.position.x:.3f}, {bone.position.y:.3f}, {bone.position.z:.3f})")
        print(f"      Anim pos: ({anim_pos.x:.3f}, {anim_pos.y:.3f}, {anim_pos.z:.3f})")
        print(f"      Anim quat: ({anim_quat.x:.3f}, {anim_quat.y:.3f}, {anim_quat.z:.3f}, {anim_quat.w:.3f})")
        print(f"      World pos: ({current_world_pos[0]:.3f}, {current_world_pos[1]:.3f}, {current_world_pos[2]:.3f})")
    
    # Step 4: Get animation data for target bone
    print(f"\n--- Animation Data for '{bone_name}' ---")
    anim_pos, anim_quat = get_bone_animation_data(vmd_motion, bone_name, frame)
    print(f"Animation position: ({anim_pos.x:.3f}, {anim_pos.y:.3f}, {anim_pos.z:.3f})")
    print(f"Animation quaternion: ({anim_quat.x:.3f}, {anim_quat.y:.3f}, {anim_quat.z:.3f}, {anim_quat.w:.3f})")
    
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
    
    # Step 6: Calculate final world position
    print(f"\n--- World Position Calculation ---")
    try:
        world_pos = get_bone_world_position(pmx_model, vmd_motion, frame, bone_name)
        print(f"Final world position: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})")
    except Exception as e:
        print(f"ERROR calculating world position: {e}")


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
    
    # Also read using direct PMX methods for comparison
    print(f"\n=== READING PMX WITH BUILT-IN PYMESHIO METHODS ===")
    
    try:
        # Read the PMX file directly using pymeshio.pmx.reader
        pmx_data_direct = pmx_reader.read_from_file(pmx_path)
        print(f"Direct PMX read successful!")
        print(f"Model name: {pmx_data_direct.name}")
        print(f"Total bones (direct method): {len(pmx_data_direct.bones)}")
    except Exception as e:
        print(f"Direct PMX read failed: {e}")
        pmx_data_direct = None
    
    # Print all bone initial positions - Method 1 (current approach)
    print(f"\n=== METHOD 1: ALL BONE INITIAL POSITIONS (pmx_reader) ===")
    print(f"Total bones: {len(pmx_model.bones)}")
    
    for i, bone in enumerate(pmx_model.bones):
        print(f"{i:3d}: {bone.name:25s} pos=({bone.position.x:8.3f}, {bone.position.y:8.3f}, {bone.position.z:8.3f}) parent={bone.parent_index}")
    
    # Print all bone initial positions - Method 2 (direct pymeshio.pmx)
    if pmx_data_direct:
        print(f"\n=== METHOD 2: ALL BONE INITIAL POSITIONS (direct pymeshio.pmx) ===")
        print(f"Total bones: {len(pmx_data_direct.bones)}")
        
        for i, bone in enumerate(pmx_data_direct.bones):
            print(f"{i:3d}: {bone.name:25s} pos=({bone.position.x:8.3f}, {bone.position.y:8.3f}, {bone.position.z:8.3f}) parent={bone.parent_index}")
        
        # Compare key bones between methods
        print(f"\n=== COMPARISON BETWEEN METHODS ===")
        target_bones = ['左足首', '右足首', '左つま先', '右つま先']  # ankle and toe bones
        
        for bone_name in target_bones:
            # Find in method 1
            bone1 = None
            for i, bone in enumerate(pmx_model.bones):
                if bone.name == bone_name:
                    bone1 = (i, bone)
                    break
            
            # Find in method 2
            bone2 = None
            for i, bone in enumerate(pmx_data_direct.bones):
                if bone.name == bone_name:
                    bone2 = (i, bone)
                    break
            
            if bone1 and bone2:
                idx1, b1 = bone1
                idx2, b2 = bone2
                print(f"{bone_name}:")
                print(f"  Method 1: idx={idx1}, pos=({b1.position.x:.3f}, {b1.position.y:.3f}, {b1.position.z:.3f})")
                print(f"  Method 2: idx={idx2}, pos=({b2.position.x:.3f}, {b2.position.y:.3f}, {b2.position.z:.3f})")
                
                pos_diff = ((b1.position.x - b2.position.x)**2 +
                           (b1.position.y - b2.position.y)**2 +
                           (b1.position.z - b2.position.z)**2) ** 0.5
                print(f"  Position difference: {pos_diff:.6f}")
    
    # Show that both methods are actually the same - just for clarity
    print(f"\n=== VERIFICATION ===")
    print(f"pmx_model is pmx_data_direct: {pmx_model is pmx_data_direct}")
    print(f"Same object references: {pmx_model == pmx_data_direct}")
    
    frame = 3000
    
    # Debug both left and right foot/ankle bones
    debug_bone_detailed(pmx_model, vmd_motion, frame, '左足首')  # left ankle
    debug_bone_detailed(pmx_model, vmd_motion, frame, '右足首')  # right ankle
    debug_bone_detailed(pmx_model, vmd_motion, frame, '左つま先')  # left toe
    debug_bone_detailed(pmx_model, vmd_motion, frame, '右つま先')  # right toe
    
    print(f"\n=== COMPARISON ===")
    left_ankle_pos = get_bone_world_position(pmx_model, vmd_motion, frame, '左足首')
    right_ankle_pos = get_bone_world_position(pmx_model, vmd_motion, frame, '右足首')
    left_toe_pos = get_bone_world_position(pmx_model, vmd_motion, frame, '左つま先')
    right_toe_pos = get_bone_world_position(pmx_model, vmd_motion, frame, '右つま先')
    
    print(f"LEFT_ANKLE:  ({left_ankle_pos[0]:8.3f}, {left_ankle_pos[1]:8.3f}, {left_ankle_pos[2]:8.3f})")
    print(f"RIGHT_ANKLE: ({right_ankle_pos[0]:8.3f}, {right_ankle_pos[1]:8.3f}, {right_ankle_pos[2]:8.3f})")
    print(f"LEFT_TOE:    ({left_toe_pos[0]:8.3f}, {left_toe_pos[1]:8.3f}, {left_toe_pos[2]:8.3f})")
    print(f"RIGHT_TOE:   ({right_toe_pos[0]:8.3f}, {right_toe_pos[1]:8.3f}, {right_toe_pos[2]:8.3f})")
    
    ankle_y_diff = abs(left_ankle_pos[1] - right_ankle_pos[1])
    toe_y_diff = abs(left_toe_pos[1] - right_toe_pos[1])
    
    print(f"\nAnkle Y difference: {ankle_y_diff:.6f}")
    print(f"Toe Y difference: {toe_y_diff:.6f}")
    
    if ankle_y_diff < 0.001:
        print("⚠️  PROBLEM: Identical ankle Y coordinates!")
    else:
        print("✅ Ankle Y coordinates are different - good!")
        
    if toe_y_diff < 0.001:
        print("⚠️  PROBLEM: Identical toe Y coordinates!")
    else:
        print("✅ Toe Y coordinates are different - good!")

if __name__ == "__main__":
    main()