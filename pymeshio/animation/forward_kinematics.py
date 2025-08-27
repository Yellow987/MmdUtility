# coding: utf-8
"""
PyTorch-based forward kinematics for MMD bone animations.

Provides efficient tensor-based calculations for computing world positions
of bones given their rest poses, rotations, and parent-child relationships.
"""

import torch
from typing import List, Optional


class ForwardKinematics:
    """PyTorch-based forward kinematics calculator for bone animations."""
    
    def __init__(self, device='cpu'):
        """
        Initialize the forward kinematics calculator.
        
        Args:
            device: PyTorch device to use for calculations ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
    
    def compute_world_positions(self,
                              bone_offsets: torch.Tensor,
                              quaternions: torch.Tensor,
                              parent_indices: List[int],
                              local_translations: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute world positions using forward kinematics.
        
        Args:
            bone_offsets: [N, 3] tensor of bone offsets from parent (bind pose)
            quaternions: [N, 4] tensor of local rotations (x,y,z,w) - normalized
            parent_indices: List of parent bone indices (-1 for root)
            local_translations: [N, 3] tensor of per-frame local translations (optional)
            
        Returns:
            world_positions: [N, 3] tensor of world positions
        """
        n_bones = bone_offsets.shape[0]
        world_positions = torch.zeros_like(bone_offsets)
        world_rotations = quaternions.clone()  # Start with normalized quaternions
        
        if local_translations is None:
            local_translations = torch.zeros_like(bone_offsets)
        
        # Get traversal order (parents before children)
        traversal_order = self._get_traversal_order(parent_indices)
        
        for j in traversal_order:
            p = parent_indices[j]
            
            if p == -1:  # Root bone
                world_positions[j] = local_translations[j]
                world_rotations[j] = quaternions[j]
            else:  # Child bone
                parent_rot = world_rotations[p]
                world_rotations[j] = self._quaternion_multiply(parent_rot, quaternions[j])
                step = bone_offsets[j] + local_translations[j]
                world_positions[j] = world_positions[p] + self._rotate_vector_by_quaternion(step, parent_rot)
        
        return world_positions
    
    def _get_traversal_order(self, parent_indices: List[int]) -> List[int]:
        """Get bone traversal order (topological sort - parents before children)."""
        n_bones = len(parent_indices)
        visited = [False] * n_bones
        order = []
        
        def visit(bone_idx):
            if visited[bone_idx]:
                return
            
            parent_idx = parent_indices[bone_idx]
            if parent_idx != -1:
                visit(parent_idx)  # Visit parent first
            
            visited[bone_idx] = True
            order.append(bone_idx)
        
        for i in range(n_bones):
            visit(i)
        
        return order
    
    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions: q1 * q2."""
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        
        result = torch.empty_like(q1)
        result[0] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        result[1] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        result[2] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        result[3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return result
    
    def _rotate_vector_by_quaternion(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q using efficient method."""
        q_vec = q[:3]  # x, y, z components
        q_w = q[3]     # w component
        
        # First cross product: cross(q_vec, v) + q_w * v
        cross1 = torch.linalg.cross(q_vec, v) + q_w * v
        
        # Second cross product: cross(q_vec, cross1)
        cross2 = torch.linalg.cross(q_vec, cross1)
        
        # Final result: v + 2 * cross2
        return v + 2.0 * cross2