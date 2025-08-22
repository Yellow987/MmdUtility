#!/usr/bin/env python
# coding: utf-8
"""
Bone Animation Module using Pytransform3d
=========================================

Alternative implementation of bone animation calculations using the pytransform3d library
for transformation matrices and quaternion handling. This provides validation against
the numpy-based implementation in bone_animation.py.

Key Features:
- Uses pytransform3d.Transform for cleaner transformation operations
- Proper quaternion SLERP interpolation instead of linear interpolation
- Built-in numerical stability and validation
- Side-by-side comparison with numpy implementation
"""

import math
import numpy as np
from typing import Tuple, Optional, Union, List

# Pytransform3d imports
from pytransform3d.transformations import transform_from, concat, transform, vectors_to_points
from pytransform3d.rotations import matrix_from_quaternion, quaternion_slerp

# Existing pymeshio imports
import pymeshio
from pymeshio import common, pmx, vmd

# Import existing functionality we'll reuse
from bone_animation import (
    BoneHierarchyWalker, 
    VMDInterpolator, 
    get_bone_animation_data,
    BoneNotFoundError,
    InvalidFrameError
)


class Pytransform3dBoneAnimator:
    """Bone animation calculator using pytransform3d library."""
    
    @staticmethod
    def create_transform_from_pos_quat(pos: common.Vector3, quat: common.Quaternion) -> np.ndarray:
        """
        Create 4x4 transformation matrix from position and quaternion using pytransform3d.
        
        Args:
            pos: Translation vector
            quat: Rotation quaternion (x,y,z,w format)
            
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        # Convert quaternion to rotation matrix
        # Note: pytransform3d expects (w,x,y,z) order
        q_array = np.array([quat.w, quat.x, quat.y, quat.z])
        rotation_matrix = matrix_from_quaternion(q_array)
        
        # Create position vector
        position = np.array([pos.x, pos.y, pos.z])
        
        # Create transform matrix using pytransform3d
        transform_matrix = transform_from(R=rotation_matrix, p=position)
        
        return transform_matrix
    
    @staticmethod
    def interpolate_bone_frame_slerp(frame1: vmd.BoneFrame, frame2: vmd.BoneFrame, 
                                   target_frame: int) -> Tuple[common.Vector3, common.Quaternion]:
        """
        Interpolate between two bone keyframes using proper quaternion SLERP.
        
        Args:
            frame1: Earlier keyframe
            frame2: Later keyframe  
            target_frame: Target frame number
            
        Returns:
            tuple: (interpolated_position, interpolated_quaternion)
        """
        if frame1.frame == frame2.frame:
            return frame1.pos, frame1.q
        
        # Calculate interpolation parameter
        t = (target_frame - frame1.frame) / (frame2.frame - frame1.frame)
        t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
        
        # Parse Bezier control points (reuse existing logic)
        bezier_data = VMDInterpolator.parse_complement_data(frame2.complement)
        
        # Interpolate position using existing Bezier logic
        t_x = VMDInterpolator.bezier_interpolate(t, bezier_data['x'])
        t_y = VMDInterpolator.bezier_interpolate(t, bezier_data['y'])
        t_z = VMDInterpolator.bezier_interpolate(t, bezier_data['z'])
        
        pos = common.Vector3(
            frame1.pos.x + (frame2.pos.x - frame1.pos.x) * t_x,
            frame1.pos.y + (frame2.pos.y - frame1.pos.y) * t_y,
            frame1.pos.z + (frame2.pos.z - frame1.pos.z) * t_z
        )
        
        # Improved quaternion interpolation using SLERP
        t_rot = VMDInterpolator.bezier_interpolate(t, bezier_data['rot'])
        
        # Convert to pytransform3d quaternion format (w,x,y,z)
        q1 = np.array([frame1.q.w, frame1.q.x, frame1.q.y, frame1.q.z])
        q2 = np.array([frame2.q.w, frame2.q.x, frame2.q.y, frame2.q.z])
        
        # Use pytransform3d's SLERP
        interpolated_q = quaternion_slerp(q1, q2, t_rot)
        
        # Convert back to common.Quaternion (x,y,z,w format)
        quat = common.Quaternion(
            interpolated_q[1],  # x
            interpolated_q[2],  # y
            interpolated_q[3],  # z
            interpolated_q[0]   # w
        )
        
        return pos, quat


def get_bone_animation_data_slerp(vmd_motion: vmd.Motion, bone_name: Union[str, bytes], 
                                frame_number: int) -> Tuple[common.Vector3, common.Quaternion]:
    """
    Get bone animation data using improved SLERP interpolation.
    
    Args:
        vmd_motion: VMD motion data
        bone_name: Name of the bone
        frame_number: Target frame number
        
    Returns:
        tuple: (position, quaternion) for the bone at the given frame
    """
    # Handle encoding (reuse existing logic)
    if isinstance(bone_name, str):
        try:
            bone_name = bone_name.encode('shift-jis')
        except UnicodeEncodeError:
            try:
                bone_name = bone_name.encode('utf-8')
            except UnicodeEncodeError:
                bone_name = bone_name.encode('utf-8', errors='replace')
    
    # Find all keyframes for this bone
    bone_frames = []
    for frame in vmd_motion.motions:
        frame_name = frame.name
        if isinstance(frame_name, str):
            frame_name = frame_name.encode('utf-8')
        
        if frame_name == bone_name:
            bone_frames.append(frame)
    
    if not bone_frames:
        # No animation data for this bone, return identity transform
        return common.Vector3(0, 0, 0), common.Quaternion(0, 0, 0, 1)
    
    # Sort frames by frame number
    bone_frames.sort(key=lambda f: f.frame)
    
    # Find keyframes to interpolate between
    if frame_number <= bone_frames[0].frame:
        # Before first keyframe
        return bone_frames[0].pos, bone_frames[0].q
    elif frame_number >= bone_frames[-1].frame:
        # After last keyframe
        return bone_frames[-1].pos, bone_frames[-1].q
    else:
        # Find surrounding keyframes
        prev_frame = bone_frames[0]
        next_frame = bone_frames[-1]
        
        for frame in bone_frames:
            if frame.frame <= frame_number:
                prev_frame = frame
            if frame.frame >= frame_number:
                next_frame = frame
                break
        
        if prev_frame.frame == next_frame.frame:
            return prev_frame.pos, prev_frame.q
        
        # Use improved SLERP interpolation
        return Pytransform3dBoneAnimator.interpolate_bone_frame_slerp(
            prev_frame, next_frame, frame_number
        )


def get_bone_world_position_pt3d(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                               frame_number: int, bone_name: Union[str, bytes]) -> Tuple[float, float, float]:
    """
    Calculate world space position of a bone using pytransform3d.
    
    This is the alternative implementation to get_bone_world_position() that uses
    pytransform3d for all transformation operations, providing validation and
    potentially better numerical stability.
    
    Args:
        pmx_model: PMX Model object containing bone hierarchy
        vmd_motion: VMD Motion object containing animation keyframes
        frame_number: Target frame number (int)
        bone_name: Name of the bone (str or bytes)
        
    Returns:
        tuple: (x, y, z) world position coordinates
        
    Raises:
        BoneNotFoundError: If bone doesn't exist in model
        InvalidFrameError: If frame number is invalid
    """
    # Validate inputs
    if frame_number < 0:
        raise InvalidFrameError(f"Frame number must be non-negative, got {frame_number}")
    
    # Find target bone (reuse existing logic)
    bone_result = BoneHierarchyWalker.find_bone_by_name(pmx_model.bones, bone_name)
    if bone_result is None:
        raise BoneNotFoundError(f"Bone '{bone_name}' not found in PMX model")
    
    bone_index, target_bone = bone_result
    
    # Get bone chain from root to target (reuse existing logic)
    bone_chain = BoneHierarchyWalker.get_bone_chain_to_root(pmx_model.bones, bone_index)
    
    # Start with identity transformation matrix
    world_transform = np.eye(4)
    
    # Apply transformations from root to target
    for chain_bone_index in bone_chain:
        bone = pmx_model.bones[chain_bone_index]
        
        # Calculate local position (same logic as numpy version)
        if bone.parent_index == -1:
            # Root bone - local position equals world position
            local_pos = bone.position
        else:
            # Child bone - local position = bone_world_pos - parent_world_pos
            parent_bone = pmx_model.bones[bone.parent_index]
            local_pos = common.Vector3(
                bone.position.x - parent_bone.position.x,
                bone.position.y - parent_bone.position.y,
                bone.position.z - parent_bone.position.z
            )
        
        # Get animation data with improved SLERP interpolation
        anim_pos, anim_quat = get_bone_animation_data_slerp(vmd_motion, bone.name, frame_number)
        
        # Create transformations using pytransform3d functions
        
        # 1. Rest translation (local bone position)
        rest_pos = np.array([local_pos.x, local_pos.y, local_pos.z])
        rest_transform = transform_from(R=np.eye(3), p=rest_pos)
        
        # 2. Rotation from animation quaternion
        q_array = np.array([anim_quat.w, anim_quat.x, anim_quat.y, anim_quat.z])
        rotation_matrix = matrix_from_quaternion(q_array)
        rotation_transform = transform_from(R=rotation_matrix, p=np.zeros(3))
        
        # 3. Animation translation
        anim_pos_array = np.array([anim_pos.x, anim_pos.y, anim_pos.z])
        anim_transform = transform_from(R=np.eye(3), p=anim_pos_array)
        
        # Combine transformations: Rest * Rotation * Animation
        # Use pytransform3d's concat function for proper matrix multiplication
        bone_transform = concat(rest_transform, rotation_transform, anim_transform)
        
        # Accumulate transformation
        world_transform = concat(world_transform, bone_transform)
    
    # Extract world position from final transformation matrix
    # The world position is in the translation component
    world_pos = world_transform[:3, 3]
    
    return float(world_pos[0]), float(world_pos[1]), float(world_pos[2])


def compare_implementations(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                          frame_number: int, bone_name: Union[str, bytes]) -> dict:
    """
    Compare numpy vs pytransform3d implementations side by side.
    
    Args:
        pmx_model: PMX Model object
        vmd_motion: VMD Motion object
        frame_number: Target frame number
        bone_name: Bone name to test
        
    Returns:
        dict: Comparison results with both outputs and difference metrics
    """
    from bone_animation import get_bone_world_position  # Import original function
    
    results = {
        'bone_name': bone_name,
        'frame_number': frame_number,
        'numpy_result': None,
        'pytransform3d_result': None,
        'numpy_error': None,
        'pytransform3d_error': None,
        'difference': None,
        'distance_difference': None,
        'tolerance_check': None
    }
    
    # Test numpy implementation
    try:
        results['numpy_result'] = get_bone_world_position(pmx_model, vmd_motion, frame_number, bone_name)
    except Exception as e:
        results['numpy_error'] = str(e)
    
    # Test pytransform3d implementation
    try:
        results['pytransform3d_result'] = get_bone_world_position_pt3d(pmx_model, vmd_motion, frame_number, bone_name)
    except Exception as e:
        results['pytransform3d_error'] = str(e)
    
    # Calculate differences if both succeeded
    if results['numpy_result'] is not None and results['pytransform3d_result'] is not None:
        diff = np.array(results['numpy_result']) - np.array(results['pytransform3d_result'])
        results['difference'] = tuple(diff)
        results['distance_difference'] = float(np.linalg.norm(diff))
        results['tolerance_check'] = results['distance_difference'] < 1e-6
    
    return results


def get_foot_positions_pt3d(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                          frame_number: int) -> dict:
    """
    Get positions of common foot bones using pytransform3d implementation.
    
    Args:
        pmx_model: PMX Model object
        vmd_motion: VMD Motion object  
        frame_number: Target frame number
        
    Returns:
        dict: Dictionary with foot bone positions, or None if bone not found
    """
    from bone_animation import COMMON_FOOT_BONE_NAMES
    
    results = {}
    
    for foot_type, bone_names in COMMON_FOOT_BONE_NAMES.items():
        results[foot_type] = None
        
        for bone_name in bone_names:
            try:
                pos = get_bone_world_position_pt3d(pmx_model, vmd_motion, frame_number, bone_name)
                results[foot_type] = pos
                break  # Found the bone, stop trying other names
            except BoneNotFoundError:
                continue  # Try next bone name
    
    return results


# Export main functions
__all__ = [
    'get_bone_world_position_pt3d',
    'get_bone_animation_data_slerp', 
    'compare_implementations',
    'get_foot_positions_pt3d',
    'Pytransform3dBoneAnimator'
]