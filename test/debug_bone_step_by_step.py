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

# Import pytransform3d alternative implementation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bone_animation_pytransform3d import (
    get_bone_world_position_pt3d,
    get_bone_animation_data_slerp,
    compare_implementations
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
        
        # Calculate local position (relative to parent) - same logic as in get_bone_world_position
        if bone.parent_index == -1:
            # Root bone - local position equals world position
            local_pos_x, local_pos_y, local_pos_z = bone.position.x, bone.position.y, bone.position.z
        else:
            # Child bone - local position = bone_world_pos - parent_world_pos
            parent_bone = pmx_model.bones[bone.parent_index]
            local_pos_x = bone.position.x - parent_bone.position.x
            local_pos_y = bone.position.y - parent_bone.position.y
            local_pos_z = bone.position.z - parent_bone.position.z
        
        # Create translation matrix for local rest position
        rest_translation = np.eye(4, dtype=float)
        rest_translation[0:3, 3] = [local_pos_x, local_pos_y, local_pos_z]
        
        # Create translation matrix for animation position
        anim_translation = np.eye(4, dtype=float)
        anim_translation[0:3, 3] = [anim_pos.x, anim_pos.y, anim_pos.z]
        
        # Get rotation matrix from quaternion
        rotation_matrix = anim_quat.getMatrix()
        
        # Correct transformation order: Rest_Translation * Rotation * Anim_Translation
        bone_transform = np.dot(np.dot(rest_translation, rotation_matrix), anim_translation)
        
        # Accumulate transformation
        world_transform = np.dot(world_transform, bone_transform)
        
        # Extract current world position
        current_world_pos = world_transform[0:3, 3]
        
        print(f"  {i}: Index {chain_bone_index} - '{bone.name}' (parent: {bone.parent_index})")
        print(f"      Rest pos (world): ({bone.position.x:.3f}, {bone.position.y:.3f}, {bone.position.z:.3f})")
        print(f"      Local pos: ({local_pos_x:.3f}, {local_pos_y:.3f}, {local_pos_z:.3f})")
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


def debug_bone_comparison(pmx_model, vmd_motion, frame, bone_name):
    """Compare both numpy and pytransform3d implementations side by side"""
    
    print(f"\n--- COMPARING IMPLEMENTATIONS FOR '{bone_name}' AT FRAME {frame} ---")
    
    # Get comparison results
    comparison = compare_implementations(pmx_model, vmd_motion, frame, bone_name)
    
    # Display results
    print(f"Bone: {comparison['bone_name']}")
    print(f"Frame: {comparison['frame_number']}")
    print()
    
    if comparison['numpy_error']:
        print(f"❌ NumPy implementation failed: {comparison['numpy_error']}")
    else:
        numpy_pos = comparison['numpy_result']
        print(f"✅ NumPy result:        ({numpy_pos[0]:10.6f}, {numpy_pos[1]:10.6f}, {numpy_pos[2]:10.6f})")
    
    if comparison['pytransform3d_error']:
        print(f"❌ Pytransform3d implementation failed: {comparison['pytransform3d_error']}")
    else:
        pt3d_pos = comparison['pytransform3d_result']
        print(f"✅ Pytransform3d result: ({pt3d_pos[0]:10.6f}, {pt3d_pos[1]:10.6f}, {pt3d_pos[2]:10.6f})")
    
    # Show comparison if both succeeded
    if comparison['difference'] is not None:
        diff = comparison['difference']
        print(f"📊 Difference:          ({diff[0]:10.6f}, {diff[1]:10.6f}, {diff[2]:10.6f})")
        print(f"📏 Distance difference:  {comparison['distance_difference']:.10f}")
        
        if comparison['tolerance_check']:
            print("✅ Results match within tolerance (< 1e-6)!")
        else:
            print("⚠️  Results differ beyond tolerance!")
            
        # Calculate relative error
        if comparison['numpy_result'] is not None:
            numpy_magnitude = np.linalg.norm(comparison['numpy_result'])
            if numpy_magnitude > 1e-10:
                relative_error = comparison['distance_difference'] / numpy_magnitude
                print(f"📈 Relative error:       {relative_error:.10f} ({relative_error*100:.8f}%)")
    
    print("-" * 80)


def test_interpolation_methods(pmx_model, vmd_motion, frame, bone_name):
    """Test different interpolation methods (linear vs SLERP)"""
    
    print(f"\n--- INTERPOLATION COMPARISON FOR '{bone_name}' AT FRAME {frame} ---")
    
    # Test original linear interpolation
    try:
        anim_pos_linear, anim_quat_linear = get_bone_animation_data(vmd_motion, bone_name, frame)
        print(f"Linear interpolation:")
        print(f"  Position:   ({anim_pos_linear.x:.6f}, {anim_pos_linear.y:.6f}, {anim_pos_linear.z:.6f})")
        print(f"  Quaternion: ({anim_quat_linear.x:.6f}, {anim_quat_linear.y:.6f}, {anim_quat_linear.z:.6f}, {anim_quat_linear.w:.6f})")
    except Exception as e:
        print(f"Linear interpolation failed: {e}")
        return
    
    # Test SLERP interpolation
    try:
        anim_pos_slerp, anim_quat_slerp = get_bone_animation_data_slerp(vmd_motion, bone_name, frame)
        print(f"SLERP interpolation:")
        print(f"  Position:   ({anim_pos_slerp.x:.6f}, {anim_pos_slerp.y:.6f}, {anim_pos_slerp.z:.6f})")
        print(f"  Quaternion: ({anim_quat_slerp.x:.6f}, {anim_quat_slerp.y:.6f}, {anim_quat_slerp.z:.6f}, {anim_quat_slerp.w:.6f})")
        
        # Calculate quaternion difference
        q_diff = np.array([
            anim_quat_linear.x - anim_quat_slerp.x,
            anim_quat_linear.y - anim_quat_slerp.y,
            anim_quat_linear.z - anim_quat_slerp.z,
            anim_quat_linear.w - anim_quat_slerp.w
        ])
        q_distance = np.linalg.norm(q_diff)
        print(f"  Quaternion distance: {q_distance:.10f}")
        
        if q_distance < 1e-6:
            print("✅ Quaternion interpolations match closely")
        else:
            print("📊 Quaternion interpolations differ - SLERP may be more accurate")
            
    except Exception as e:
        print(f"SLERP interpolation failed: {e}")
    
    print("-" * 80)


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
    
    # Test bones to analyze
    test_bones = ['左足首', '右足首', '左つま先', '右つま先']  # left ankle, right ankle, left toe, right toe
    
    print(f"\n" + "="*100)
    print(f"IMPLEMENTATION COMPARISON TEST - FRAME {frame}")
    print(f"="*100)
    
    # Compare implementations for each bone
    for bone_name in test_bones:
        print(f"\n{'='*20} {bone_name} {'='*20}")
        
        # Run detailed debug for numpy implementation
        debug_bone_detailed(pmx_model, vmd_motion, frame, bone_name)
        
        # Compare both implementations
        debug_bone_comparison(pmx_model, vmd_motion, frame, bone_name)
        
        # Test interpolation methods
        test_interpolation_methods(pmx_model, vmd_motion, frame, bone_name)
    
    # Summary comparison
    print(f"\n" + "="*100)
    print(f"SUMMARY COMPARISON - FRAME {frame}")
    print(f"="*100)
    
    numpy_positions = {}
    pt3d_positions = {}
    
    for bone_name in test_bones:
        try:
            numpy_pos = get_bone_world_position(pmx_model, vmd_motion, frame, bone_name)
            numpy_positions[bone_name] = numpy_pos
        except Exception as e:
            numpy_positions[bone_name] = f"ERROR: {e}"
            
        try:
            pt3d_pos = get_bone_world_position_pt3d(pmx_model, vmd_motion, frame, bone_name)
            pt3d_positions[bone_name] = pt3d_pos
        except Exception as e:
            pt3d_positions[bone_name] = f"ERROR: {e}"
    
    print(f"\nNumPy Results:")
    for bone_name in test_bones:
        pos = numpy_positions[bone_name]
        if isinstance(pos, tuple):
            print(f"  {bone_name:12s}: ({pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f})")
        else:
            print(f"  {bone_name:12s}: {pos}")
    
    print(f"\nPytransform3d Results:")
    for bone_name in test_bones:
        pos = pt3d_positions[bone_name]
        if isinstance(pos, tuple):
            print(f"  {bone_name:12s}: ({pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f})")
        else:
            print(f"  {bone_name:12s}: {pos}")
    
    # Calculate differences
    print(f"\nDifferences (NumPy - Pytransform3d):")
    overall_max_diff = 0.0
    successful_comparisons = 0
    
    for bone_name in test_bones:
        numpy_pos = numpy_positions[bone_name]
        pt3d_pos = pt3d_positions[bone_name]
        
        if isinstance(numpy_pos, tuple) and isinstance(pt3d_pos, tuple):
            diff = np.array(numpy_pos) - np.array(pt3d_pos)
            distance = np.linalg.norm(diff)
            print(f"  {bone_name:12s}: ({diff[0]:8.6f}, {diff[1]:8.6f}, {diff[2]:8.6f}) | Distance: {distance:.8f}")
            overall_max_diff = max(overall_max_diff, distance)
            successful_comparisons += 1
        else:
            print(f"  {bone_name:12s}: Cannot compare - one implementation failed")
    
    # Overall assessment
    print(f"\n" + "-"*60)
    print(f"OVERALL ASSESSMENT:")
    print(f"  Successful comparisons: {successful_comparisons}/{len(test_bones)}")
    print(f"  Maximum difference: {overall_max_diff:.10f}")
    
    if overall_max_diff < 1e-6:
        print(f"  ✅ All results match within tolerance!")
    elif overall_max_diff < 1e-3:
        print(f"  ⚠️  Small differences detected - investigate further")
    else:
        print(f"  ❌ Significant differences - implementation issue likely")
        
    # Traditional Y-difference check for validation
    if isinstance(numpy_positions.get('左足首'), tuple) and isinstance(numpy_positions.get('右足首'), tuple):
        ankle_y_diff = abs(numpy_positions['左足首'][1] - numpy_positions['右足首'][1])
        print(f"  Ankle Y difference (validation): {ankle_y_diff:.6f}")
        
        if ankle_y_diff < 0.001:
            print("  ⚠️  PROBLEM: Identical ankle Y coordinates!")
        else:
            print("  ✅ Ankle Y coordinates are different - good!")

if __name__ == "__main__":
    main()