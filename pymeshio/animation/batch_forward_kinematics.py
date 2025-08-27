# coding: utf-8
"""
Batch PyTorch-based forward kinematics for MMD bone animations.

Provides efficient batch tensor-based calculations for computing world positions
of multiple frames simultaneously, with support for bone filtering.
"""

import torch
from typing import List, Optional, Set
import numpy as np


class BatchForwardKinematics:
    """Batch PyTorch-based forward kinematics calculator for bone animations."""
    
    def __init__(self, device='cpu'):
        """
        Initialize the batch forward kinematics calculator.
        
        Args:
            device: PyTorch device to use for calculations ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
    
    def compute_world_positions_batch(self,
                                    bone_offsets: torch.Tensor,
                                    quaternions_batch: torch.Tensor,
                                    parent_indices: List[int],
                                    local_translations_batch: Optional[torch.Tensor] = None,
                                    bone_filter_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Compute world positions for multiple frames using batch forward kinematics.
        
        Args:
            bone_offsets: [N, 3] tensor of bone offsets from parent (bind pose)
            quaternions_batch: [F, N, 4] tensor of local rotations for F frames (x,y,z,w) - normalized
            parent_indices: List of parent bone indices (-1 for root)
            local_translations_batch: [F, N, 3] tensor of per-frame local translations (optional)
            bone_filter_indices: List of bone indices to compute (optional, computes all if None)
            
        Returns:
            world_positions_batch: [F, M, 3] tensor of world positions (M = filtered bones or N)
        """
        n_frames, n_bones = quaternions_batch.shape[:2]
        
        # Apply bone filtering if specified
        if bone_filter_indices is not None:
            bone_indices_set = set(bone_filter_indices)
            # We need to compute all bones up to the filtered ones due to hierarchy
            max_filtered_idx = max(bone_filter_indices)
            compute_indices = list(range(max_filtered_idx + 1))
            
            # Filter tensors for computation
            compute_bone_offsets = bone_offsets[:max_filtered_idx + 1]
            compute_quaternions = quaternions_batch[:, :max_filtered_idx + 1]
            compute_parent_indices = parent_indices[:max_filtered_idx + 1]
            
            if local_translations_batch is not None:
                compute_local_translations = local_translations_batch[:, :max_filtered_idx + 1]
            else:
                compute_local_translations = None
        else:
            bone_indices_set = set(range(n_bones))
            compute_bone_offsets = bone_offsets
            compute_quaternions = quaternions_batch
            compute_parent_indices = parent_indices
            compute_local_translations = local_translations_batch
        
        n_compute_bones = compute_bone_offsets.shape[0]
        world_positions_batch = torch.zeros(n_frames, n_compute_bones, 3, 
                                          dtype=torch.float32, device=self.device)
        world_rotations_batch = compute_quaternions.clone()
        
        if compute_local_translations is None:
            compute_local_translations = torch.zeros(n_frames, n_compute_bones, 3, 
                                                   dtype=torch.float32, device=self.device)
        
        # Get traversal order (parents before children)
        traversal_order = self._get_traversal_order(compute_parent_indices)
        
        # Process each bone in dependency order (vectorized across frames)
        for j in traversal_order:
            p = compute_parent_indices[j]
            
            if p == -1:  # Root bone
                world_positions_batch[:, j] = compute_local_translations[:, j]
                world_rotations_batch[:, j] = compute_quaternions[:, j]
            else:  # Child bone
                # Batch quaternion multiplication: parent_rot * local_rot
                parent_rot_batch = world_rotations_batch[:, p]  # [F, 4]
                local_rot_batch = compute_quaternions[:, j]     # [F, 4]
                world_rotations_batch[:, j] = self._quaternion_multiply_batch(parent_rot_batch, local_rot_batch)
                
                # Batch vector rotation and translation
                step_batch = compute_bone_offsets[j].unsqueeze(0) + compute_local_translations[:, j]  # [F, 3]
                rotated_step_batch = self._rotate_vector_by_quaternion_batch(step_batch, parent_rot_batch)
                world_positions_batch[:, j] = world_positions_batch[:, p] + rotated_step_batch
        
        # Filter output to only requested bones
        if bone_filter_indices is not None:
            # Create output tensor for filtered bones only
            filtered_positions = torch.zeros(n_frames, len(bone_filter_indices), 3,
                                           dtype=torch.float32, device=self.device)
            for out_idx, bone_idx in enumerate(bone_filter_indices):
                if bone_idx < n_compute_bones:
                    filtered_positions[:, out_idx] = world_positions_batch[:, bone_idx]
            return filtered_positions
        
        return world_positions_batch
    
    def _get_traversal_order(self, parent_indices: List[int]) -> List[int]:
        """Get bone traversal order (topological sort - parents before children)."""
        n_bones = len(parent_indices)
        visited = [False] * n_bones
        order = []
        
        def visit(bone_idx):
            if visited[bone_idx]:
                return
            
            parent_idx = parent_indices[bone_idx]
            if parent_idx != -1 and parent_idx < n_bones:
                visit(parent_idx)  # Visit parent first
            
            visited[bone_idx] = True
            order.append(bone_idx)
        
        for i in range(n_bones):
            visit(i)
        
        return order
    
    def _quaternion_multiply_batch(self, q1_batch: torch.Tensor, q2_batch: torch.Tensor) -> torch.Tensor:
        """Multiply two batches of quaternions: q1 * q2. Input shape: [F, 4]."""
        x1, y1, z1, w1 = q1_batch[:, 0], q1_batch[:, 1], q1_batch[:, 2], q1_batch[:, 3]
        x2, y2, z2, w2 = q2_batch[:, 0], q2_batch[:, 1], q2_batch[:, 2], q2_batch[:, 3]
        
        result = torch.empty_like(q1_batch)
        result[:, 0] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        result[:, 1] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        result[:, 2] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        result[:, 3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return result
    
    def _rotate_vector_by_quaternion_batch(self, v_batch: torch.Tensor, q_batch: torch.Tensor) -> torch.Tensor:
        """Rotate batch of vectors by batch of quaternions. Input shapes: [F, 3], [F, 4]."""
        q_vec = q_batch[:, :3]  # [F, 3] - x, y, z components
        q_w = q_batch[:, 3:4]   # [F, 1] - w component
        
        # Batch cross products using torch.linalg.cross
        # First cross product: cross(q_vec, v) + q_w * v
        cross1 = torch.linalg.cross(q_vec, v_batch, dim=1) + q_w * v_batch
        
        # Second cross product: cross(q_vec, cross1)
        cross2 = torch.linalg.cross(q_vec, cross1, dim=1)
        
        # Final result: v + 2 * cross2
        return v_batch + 2.0 * cross2


def create_bone_filter_indices(bone_names: List[str], filter_bone_names: List[str]) -> List[int]:
    """
    Create bone filter indices from bone names.
    
    Args:
        bone_names: List of all bone names in model order
        filter_bone_names: List of bone names to filter for
        
    Returns:
        List of bone indices corresponding to filter_bone_names
    """
    bone_name_to_index = {name: idx for idx, name in enumerate(bone_names)}
    filter_indices = []
    
    for bone_name in filter_bone_names:
        if bone_name in bone_name_to_index:
            filter_indices.append(bone_name_to_index[bone_name])
        else:
            print(f"Warning: Bone '{bone_name}' not found in model")
    
    return filter_indices