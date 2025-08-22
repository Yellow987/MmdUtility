#!/usr/bin/env python
# coding: utf-8
"""
Bone Animation Module for MmdUtility
===================================

Provides functionality to calculate world space bone positions from PMX models
and VMD animation data. Designed for machine learning applications, particularly
for dance generation and foot contact detection.

Key Features:
- Calculate world space bone positions at any frame
- Handle VMD Bezier curve interpolation between keyframes
- Support bone hierarchy traversal and transformations
- ML-friendly utilities for foot contact detection
"""

import math
import numpy as np
from typing import Tuple, Optional, Union, List
import pymeshio
from pymeshio import common, pmx, vmd


class BoneNotFoundError(Exception):
    """Raised when a requested bone is not found in the PMX model."""
    pass


class InvalidFrameError(Exception):
    """Raised when an invalid frame number is provided."""
    pass


class VMDInterpolator:
    """Handles VMD Bezier curve interpolation between keyframes."""
    
    @staticmethod
    def parse_complement_data(complement_str: str) -> dict:
        """
        Parse VMD complement data (64 bytes) into Bezier control points.
        
        VMD stores Bezier curves for X, Y, Z translation and Rotation interpolation.
        Each curve has 4 control points: (x1, x2, y1, y2)
        
        Args:
            complement_str: Hex string representation of 64-byte complement data
            
        Returns:
            dict: Bezier control points for each channel
        """
        if not complement_str or len(complement_str) != 128:  # 64 bytes = 128 hex chars
            # Default linear interpolation
            return {
                'x': (20, 107, 20, 107),
                'y': (20, 107, 20, 107), 
                'z': (20, 107, 20, 107),
                'rot': (20, 107, 20, 107)
            }
        
        try:
            # Convert hex string to bytes
            bytes_data = bytes.fromhex(complement_str)
            
            # Extract Bezier control points for each channel
            # VMD format: X_x1, X_y1, X_x2, X_y2, Y_x1, Y_y1, Y_x2, Y_y2, Z_x1, Z_y1, Z_x2, Z_y2, R_x1, R_y1, R_x2, R_y2
            return {
                'x': (bytes_data[0], bytes_data[4], bytes_data[8], bytes_data[12]),
                'y': (bytes_data[1], bytes_data[5], bytes_data[9], bytes_data[13]),
                'z': (bytes_data[2], bytes_data[6], bytes_data[10], bytes_data[14]),
                'rot': (bytes_data[3], bytes_data[7], bytes_data[11], bytes_data[15])
            }
        except (ValueError, IndexError):
            # Fall back to linear interpolation
            return {
                'x': (20, 107, 20, 107),
                'y': (20, 107, 20, 107),
                'z': (20, 107, 20, 107), 
                'rot': (20, 107, 20, 107)
            }
    
    @staticmethod
    def bezier_interpolate(t: float, control_points: Tuple[int, int, int, int]) -> float:
        """
        Calculate Bezier curve interpolation factor.
        
        Args:
            t: Time parameter [0.0, 1.0]
            control_points: (x1, x2, y1, y2) Bezier control points
            
        Returns:
            float: Interpolated value [0.0, 1.0]
        """
        x1, x2, y1, y2 = [cp / 127.0 for cp in control_points]
        
        # Use Newton-Raphson method to find t parameter for cubic Bezier
        # B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
        # Where P₀=(0,0), P₁=(x1,y1), P₂=(x2,y2), P₃=(1,1)
        
        def bezier_x(u):
            return 3 * (1 - u) * (1 - u) * u * x1 + 3 * (1 - u) * u * u * x2 + u * u * u
        
        def bezier_y(u):
            return 3 * (1 - u) * (1 - u) * u * y1 + 3 * (1 - u) * u * u * y2 + u * u * u
        
        # Find u such that bezier_x(u) = t
        u = t
        for _ in range(10):  # Newton-Raphson iterations
            x_val = bezier_x(u)
            if abs(x_val - t) < 1e-6:
                break
            
            # Derivative of bezier_x
            dx = 3 * (1 - u) * (1 - u) * x1 + 6 * (1 - u) * u * (x2 - x1) + 3 * u * u * (1 - x2)
            if abs(dx) > 1e-10:
                u = u - (x_val - t) / dx
            else:
                break
        
        return bezier_y(u)
    
    @staticmethod
    def interpolate_bone_frame(frame1: vmd.BoneFrame, frame2: vmd.BoneFrame, 
                              target_frame: int) -> Tuple[common.Vector3, common.Quaternion]:
        """
        Interpolate between two bone keyframes using VMD Bezier curves.
        
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
        
        # Parse Bezier control points
        bezier_data = VMDInterpolator.parse_complement_data(frame2.complement)
        
        # Interpolate position
        t_x = VMDInterpolator.bezier_interpolate(t, bezier_data['x'])
        t_y = VMDInterpolator.bezier_interpolate(t, bezier_data['y'])
        t_z = VMDInterpolator.bezier_interpolate(t, bezier_data['z'])
        
        pos = common.Vector3(
            frame1.pos.x + (frame2.pos.x - frame1.pos.x) * t_x,
            frame1.pos.y + (frame2.pos.y - frame1.pos.y) * t_y,
            frame1.pos.z + (frame2.pos.z - frame1.pos.z) * t_z
        )
        
        # Interpolate rotation using simple linear interpolation (can be improved later)
        t_rot = VMDInterpolator.bezier_interpolate(t, bezier_data['rot'])
        
        # Simple linear quaternion interpolation using existing common.Quaternion methods
        quat = common.Quaternion(
            frame1.q.x + (frame2.q.x - frame1.q.x) * t_rot,
            frame1.q.y + (frame2.q.y - frame1.q.y) * t_rot,
            frame1.q.z + (frame2.q.z - frame1.q.z) * t_rot,
            frame1.q.w + (frame2.q.w - frame1.q.w) * t_rot
        ).getNormalized()  # Use existing normalization method
        
        return pos, quat


class BoneHierarchyWalker:
    """Handles bone hierarchy traversal and world space calculations."""
    
    @staticmethod
    def find_bone_by_name(bones: List[pmx.Bone], bone_name: Union[str, bytes]) -> Optional[Tuple[int, pmx.Bone]]:
        """
        Find a bone by name in the bone list.
        
        Args:
            bones: List of PMX bones
            bone_name: Name to search for (str or bytes)
            
        Returns:
            tuple: (bone_index, bone) or None if not found
        """
        if isinstance(bone_name, str):
            bone_name = bone_name.encode('utf-8')
        
        for i, bone in enumerate(bones):
            # Try both Japanese and English names
            if (isinstance(bone.name, bytes) and bone.name == bone_name) or \
               (isinstance(bone.name, str) and bone.name.encode('utf-8') == bone_name) or \
               (isinstance(bone.english_name, bytes) and bone.english_name == bone_name) or \
               (isinstance(bone.english_name, str) and bone.english_name.encode('utf-8') == bone_name):
                return i, bone
        return None
    
    @staticmethod
    def get_bone_chain_to_root(bones: List[pmx.Bone], bone_index: int) -> List[int]:
        """
        Get the bone chain from target bone to root.
        
        Args:
            bones: List of PMX bones
            bone_index: Index of target bone
            
        Returns:
            list: Bone indices from root to target (in transformation order)
        """
        chain = []
        current_index = bone_index
        
        # Walk up the hierarchy to root
        visited = set()
        while current_index != -1 and current_index not in visited:
            if current_index >= len(bones):
                break
            visited.add(current_index)
            chain.append(current_index)
            current_index = bones[current_index].parent_index
        
        # Reverse to get root-to-target order
        chain.reverse()
        return chain
    
    @staticmethod
    def create_transformation_matrix(pos: common.Vector3, quat: common.Quaternion) -> List[List[float]]:
        """
        Create a 4x4 transformation matrix from position and quaternion.
        
        Args:
            pos: Translation vector
            quat: Rotation quaternion
            
        Returns:
            list: 4x4 transformation matrix
        """
        # Normalize quaternion
        norm = math.sqrt(quat.x*quat.x + quat.y*quat.y + quat.z*quat.z + quat.w*quat.w)
        if norm > 1e-10:
            qx, qy, qz, qw = quat.x/norm, quat.y/norm, quat.z/norm, quat.w/norm
        else:
            qx, qy, qz, qw = 0, 0, 0, 1
        
        # Create rotation matrix from quaternion
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        
        return [
            [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy), pos.x],
            [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx), pos.y],
            [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy), pos.z],
            [0, 0, 0, 1]
        ]
    
    @staticmethod
    def multiply_matrices(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """
        Multiply two 4x4 matrices.
        
        Args:
            a: First matrix
            b: Second matrix
            
        Returns:
            list: Result matrix a * b
        """
        result = [[0 for _ in range(4)] for _ in range(4)]
        
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result
    
    @staticmethod
    def apply_matrix_to_point(matrix: List[List[float]], point: common.Vector3) -> common.Vector3:
        """
        Apply transformation matrix to a point.
        
        Args:
            matrix: 4x4 transformation matrix
            point: Point to transform
            
        Returns:
            common.Vector3: Transformed point
        """
        x = matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2] * point.z + matrix[0][3]
        y = matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2] * point.z + matrix[1][3]
        z = matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2] * point.z + matrix[2][3]
        
        return common.Vector3(x, y, z)


def get_bone_animation_data(vmd_motion: vmd.Motion, bone_name: Union[str, bytes], 
                          frame_number: int) -> Tuple[common.Vector3, common.Quaternion]:
    """
    Get bone animation data (position and rotation) for a specific frame.
    
    Args:
        vmd_motion: VMD motion data
        bone_name: Name of the bone
        frame_number: Target frame number
        
    Returns:
        tuple: (position, quaternion) for the bone at the given frame
    """
    # Properly handle encoding - VMD files use Shift-JIS encoding
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
        
        # Interpolate between keyframes
        return VMDInterpolator.interpolate_bone_frame(prev_frame, next_frame, frame_number)


def get_bone_world_position(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                           frame_number: int, bone_name: Union[str, bytes]) -> Tuple[float, float, float]:
    """
    Calculate world space position of a bone at a specific frame.
    
    This is the main API function for calculating bone positions in world space.
    It handles bone hierarchy traversal, VMD keyframe interpolation, and coordinate
    space transformations.
    
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
        
    Example:
        >>> import pymeshio.pmx.reader as pmx_reader
        >>> import pymeshio.vmd.reader as vmd_reader
        >>> from bone_animation import get_bone_world_position
        >>> 
        >>> pmx_data = pmx_reader.read_from_file('model.pmx')
        >>> vmd_data = vmd_reader.read_from_file('dance.vmd')
        >>> 
        >>> # Get left foot position at frame 100
        >>> pos = get_bone_world_position(pmx_data, vmd_data, 100, '左足')
        >>> print(f"Left foot at frame 100: {pos}")
    """
    # Validate inputs
    if frame_number < 0:
        raise InvalidFrameError(f"Frame number must be non-negative, got {frame_number}")
    
    # Find target bone
    bone_result = BoneHierarchyWalker.find_bone_by_name(pmx_model.bones, bone_name)
    if bone_result is None:
        raise BoneNotFoundError(f"Bone '{bone_name}' not found in PMX model")
    
    bone_index, target_bone = bone_result
    
    # Get bone chain from root to target
    bone_chain = BoneHierarchyWalker.get_bone_chain_to_root(pmx_model.bones, bone_index)
    
    # Start with identity transformation
    world_transform = np.eye(4, dtype=float)
    
    # Apply transformations from root to target
    for chain_bone_index in bone_chain:
        bone = pmx_model.bones[chain_bone_index]
        
        # Calculate local position (relative to parent)
        # PMX bone positions are world coordinates, need to convert to local
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
        
        # Get animation data for this bone at the target frame
        anim_pos, anim_quat = get_bone_animation_data(vmd_motion, bone.name, frame_number)
        
        # Create translation matrix for local rest position
        rest_translation = np.eye(4, dtype=float)
        rest_translation[0:3, 3] = [local_pos.x, local_pos.y, local_pos.z]
        
        # Create translation matrix for animation position
        anim_translation = np.eye(4, dtype=float)
        anim_translation[0:3, 3] = [anim_pos.x, anim_pos.y, anim_pos.z]
        
        # Get rotation matrix from quaternion (using existing common.Quaternion)
        rotation_matrix = anim_quat.getMatrix()  # This returns numpy array
        
        # Correct transformation order: Rest_Translation * Rotation * Anim_Translation
        bone_transform = np.dot(np.dot(rest_translation, rotation_matrix), anim_translation)
        
        # Accumulate transformation (matrix multiplication order is important!)
        world_transform = np.dot(world_transform, bone_transform)
    
    # Extract world position from final transformation matrix
    world_pos = world_transform[0:3, 3]  # Get translation component
    
    return float(world_pos[0]), float(world_pos[1]), float(world_pos[2])


# Utility functions for common dance analysis tasks
COMMON_FOOT_BONE_NAMES = {
    'left_foot': ['左足', 'left foot', 'L_foot', 'foot_L'],
    'right_foot': ['右足', 'right foot', 'R_foot', 'foot_R'],
    'left_toe': ['左つま先', 'left toe', 'L_toe', 'toe_L'],
    'right_toe': ['右つま先', 'right toe', 'R_toe', 'toe_R']
}


def get_foot_positions(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                      frame_number: int) -> dict:
    """
    Get positions of common foot bones for contact detection.
    
    Args:
        pmx_model: PMX Model object
        vmd_motion: VMD Motion object  
        frame_number: Target frame number
        
    Returns:
        dict: Dictionary with foot bone positions, or None if bone not found
        
    Example:
        >>> positions = get_foot_positions(pmx_data, vmd_data, 100)
        >>> if positions['left_foot']:
        >>>     print(f"Left foot Y position: {positions['left_foot'][1]}")
    """
    results = {}
    
    for foot_type, bone_names in COMMON_FOOT_BONE_NAMES.items():
        results[foot_type] = None
        
        for bone_name in bone_names:
            try:
                pos = get_bone_world_position(pmx_model, vmd_motion, frame_number, bone_name)
                results[foot_type] = pos
                break  # Found the bone, stop trying other names
            except BoneNotFoundError:
                continue  # Try next bone name
    
    return results


def is_foot_on_ground(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                     frame_number: int, foot_bone_name: Union[str, bytes],
                     ground_threshold: float = 0.5) -> bool:
    """
    Determine if a foot bone is likely touching the ground.
    
    Args:
        pmx_model: PMX Model object
        vmd_motion: VMD Motion object
        frame_number: Target frame number  
        foot_bone_name: Name of foot bone to check
        ground_threshold: Y-coordinate threshold for ground contact
        
    Returns:
        bool: True if foot is likely on ground, False otherwise
    """
    try:
        pos = get_bone_world_position(pmx_model, vmd_motion, frame_number, foot_bone_name)
        return pos[1] <= ground_threshold  # Y coordinate below threshold
    except (BoneNotFoundError, InvalidFrameError):
        return False