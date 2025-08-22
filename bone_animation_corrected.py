#!/usr/bin/env python
# coding: utf-8
"""
Corrected Bone Animation Module
===============================

This module contains corrected bone transformation logic based on the analysis
of the mathematical errors found in the original implementations.

Key Fixes:
1. Proper transformation order
2. Correct coordinate system handling  
3. Fixed matrix accumulation direction
4. Proper quaternion handling

The corrected implementation should produce ground-level foot positions
and proper bone hierarchy relationships.
"""

import math
import numpy as np
from typing import Tuple, Optional, Union, List
import pymeshio
from pymeshio import common, pmx, vmd

# Import shared components from original
from bone_animation import (
    BoneHierarchyWalker, 
    VMDInterpolator, 
    get_bone_animation_data,
    BoneNotFoundError,
    InvalidFrameError
)


def get_bone_world_position_corrected(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                                    frame_number: int, bone_name: Union[str, bytes]) -> Tuple[float, float, float]:
    """
    Calculate world space position of a bone using corrected transformation logic.
    
    This function addresses the mathematical errors identified in the analysis:
    - Proper transformation order
    - Correct coordinate system handling
    - Fixed matrix accumulation
    
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
        
        # CORRECTED TRANSFORMATION ORDER:
        # Based on the analysis, the correct MMD transformation should be:
        # LocalTransform = Translation(rest + anim) * Rotation(anim)
        # This puts the bone at its rest position, adds animation offset, then applies rotation
        
        # Calculate final position: rest position + animation offset
        final_pos = common.Vector3(
            local_pos.x + anim_pos.x,
            local_pos.y + anim_pos.y,
            local_pos.z + anim_pos.z
        )
        
        # Create transformation matrix with corrected order
        # 1. Translation for final position (rest + animation)
        translation_matrix = np.eye(4, dtype=float)
        translation_matrix[0:3, 3] = [final_pos.x, final_pos.y, final_pos.z]
        
        # 2. Rotation matrix from quaternion
        rotation_matrix = anim_quat.getMatrix()  # Use existing common.Quaternion method
        
        # 3. Combine: Translation * Rotation (apply translation first, then rotation)
        bone_transform = np.dot(translation_matrix, rotation_matrix)
        
        # CORRECTED ACCUMULATION:
        # Accumulate transformation (parent applied first, then local)
        world_transform = np.dot(world_transform, bone_transform)
    
    # Extract world position from final transformation matrix
    world_pos = world_transform[0:3, 3]
    
    return float(world_pos[0]), float(world_pos[1]), float(world_pos[2])


def get_bone_world_position_corrected_v2(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                                        frame_number: int, bone_name: Union[str, bytes]) -> Tuple[float, float, float]:
    """
    Alternative corrected implementation with different transformation order.
    
    This version tests the alternative transformation order:
    Rotation * Translation (apply rotation first, then translation)
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
        
        # ALTERNATIVE TRANSFORMATION ORDER:
        # Rotation * Translation (rotation applied in local space, then translation)
        
        # 1. Rotation matrix from quaternion
        rotation_matrix = anim_quat.getMatrix()
        
        # 2. Create translation matrix for final position (rest + animation)
        final_pos = common.Vector3(
            local_pos.x + anim_pos.x,
            local_pos.y + anim_pos.y,
            local_pos.z + anim_pos.z
        )
        
        translation_matrix = np.eye(4, dtype=float)
        translation_matrix[0:3, 3] = [final_pos.x, final_pos.y, final_pos.z]
        
        # 3. Combine: Rotation * Translation
        bone_transform = np.dot(rotation_matrix, translation_matrix)
        
        # Accumulate transformation
        world_transform = np.dot(world_transform, bone_transform)
    
    # Extract world position
    world_pos = world_transform[0:3, 3]
    
    return float(world_pos[0]), float(world_pos[1]), float(world_pos[2])


def get_bone_world_position_corrected_v3(pmx_model: pmx.Model, vmd_motion: vmd.Motion, 
                                        frame_number: int, bone_name: Union[str, bytes]) -> Tuple[float, float, float]:
    """
    Third corrected implementation using standard 3D graphics transformation order.
    
    Standard transformation order: Scale * Rotation * Translation
    Since we don't have scale, this becomes: Rotation * Translation
    But applied to the final position (rest + animation offset)
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
        
        # STANDARD 3D GRAPHICS ORDER:
        # Create separate matrices and combine in standard order
        
        # 1. Rest position translation
        rest_translation = np.eye(4, dtype=float)
        rest_translation[0:3, 3] = [local_pos.x, local_pos.y, local_pos.z]
        
        # 2. Animation rotation
        rotation_matrix = anim_quat.getMatrix()
        
        # 3. Animation position translation
        anim_translation = np.eye(4, dtype=float)
        anim_translation[0:3, 3] = [anim_pos.x, anim_pos.y, anim_pos.z]
        
        # 4. Combine in standard order: Translation(anim) * Rotation * Translation(rest)
        temp_transform = np.dot(rotation_matrix, rest_translation)
        bone_transform = np.dot(anim_translation, temp_transform)
        
        # Accumulate transformation
        world_transform = np.dot(world_transform, bone_transform)
    
    # Extract world position
    world_pos = world_transform[0:3, 3]
    
    return float(world_pos[0]), float(world_pos[1]), float(world_pos[2])


# Export the corrected functions
__all__ = [
    'get_bone_world_position_corrected',
    'get_bone_world_position_corrected_v2', 
    'get_bone_world_position_corrected_v3'
]