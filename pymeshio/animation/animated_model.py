# coding: utf-8
"""
AnimatedModel class for combining PMX models with VMD motion data.

Provides a high-level interface for querying bone world positions at specific frames.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from .. import pmx as pmx_module
from .. import vmd as vmd_module
from .forward_kinematics import ForwardKinematics


class AnimatedModel:
    """
    Combines a PMX model with VMD motion data to provide frame-based bone position queries.
    
    This class handles the complexity of combining PMX skeletal data with VMD animation data,
    providing a simple interface to query world positions of all bones at any frame.
    """
    
    def __init__(self, pmx_model: Any, vmd_motion: Any, device: str = 'cpu'):
        """
        Initialize the animated model.
        
        Args:
            pmx_model: Loaded PMX model (from pymeshio.pmx.reader)
            vmd_motion: Loaded VMD motion data (from pymeshio.vmd.reader)
            device: PyTorch device for calculations ('cpu' or 'cuda')
        """
        self.pmx_model = pmx_model
        self.vmd_motion = vmd_motion
        self.device = device
        self.fk_calculator = ForwardKinematics(device=device)
        
        # Initialize data structures
        self.bone_name_to_index: Dict[str, int] = {}
        self.bone_rest_positions: Dict[str, np.ndarray] = {}
        self.parent_indices: List[int] = []
        self.bone_offsets: np.ndarray = None
        self.vmd_data_by_frame: Dict[int, Dict[str, Dict]] = {}
        
        # Cache for performance
        self._world_positions_cache: Dict[int, torch.Tensor] = {}
        
        # Process the model and motion data
        self._process_pmx_data()
        self._process_vmd_data()
    
    def _process_pmx_data(self):
        """Process PMX model data to extract bone hierarchy and rest positions."""
        if not hasattr(self.pmx_model, 'bones'):
            raise ValueError("PMX model does not contain bone data")
        
        # Build bone name to index mapping
        for i, bone in enumerate(self.pmx_model.bones):
            self.bone_name_to_index[bone.name] = i
        
        # Extract rest positions and build hierarchy
        n_bones = len(self.pmx_model.bones)
        self.parent_indices = [-1] * n_bones
        bone_positions = []
        
        for i, bone in enumerate(self.pmx_model.bones):
            # Store rest position in MMD coordinates
            rest_pos = np.array([bone.position.x, bone.position.y, bone.position.z])
            self.bone_rest_positions[bone.name] = rest_pos
            bone_positions.append(rest_pos)
            
            # Find parent index
            if hasattr(bone, 'parent_index') and bone.parent_index != -1:
                self.parent_indices[i] = bone.parent_index
        
        # Calculate relative bone offsets (child_pos - parent_pos)
        bone_offsets = []
        for i, bone in enumerate(self.pmx_model.bones):
            parent_idx = self.parent_indices[i]
            if parent_idx != -1:
                # Calculate relative offset from parent
                child_pos = bone_positions[i]
                parent_pos = bone_positions[parent_idx]
                offset = child_pos - parent_pos
            else:
                # Root bone uses absolute position
                offset = bone_positions[i]
            bone_offsets.append(offset)
        
        self.bone_offsets = np.array(bone_offsets)
    
    def _process_vmd_data(self):
        """Process VMD motion data and organize by frame."""
        if not hasattr(self.vmd_motion, 'motions'):
            raise ValueError("VMD motion does not contain bone motion data")
        
        # Organize VMD data by frame
        for bone_frame in self.vmd_motion.motions:
            frame_num = bone_frame.frame
            bone_name = bone_frame.name
            
            # Fix encoding issue: decode bytes to proper Unicode string if needed
            if isinstance(bone_name, bytes):
                try:
                    bone_name = bone_name.decode('shift_jis')
                except (UnicodeDecodeError, AttributeError):
                    try:
                        bone_name = bone_name.decode('utf-8')
                    except (UnicodeDecodeError, AttributeError):
                        # Fallback: use string representation
                        bone_name = str(bone_name)
            
            if frame_num not in self.vmd_data_by_frame:
                self.vmd_data_by_frame[frame_num] = {}
            
            # Store position and quaternion data
            self.vmd_data_by_frame[frame_num][bone_name] = {
                'position': np.array([bone_frame.pos.x, bone_frame.pos.y, bone_frame.pos.z]),
                'quaternion': np.array([bone_frame.q.x, bone_frame.q.y, bone_frame.q.z, bone_frame.q.w])
            }
    
    def get_world_positions(self, frame: int, use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        Get world positions of all bones at the specified frame.
        
        Args:
            frame: Frame number to query
            use_cache: Whether to use cached results for performance
            
        Returns:
            Dictionary mapping bone_name -> world_position (as numpy array in MMD coordinates)
        """
        if use_cache and frame in self._world_positions_cache:
            cached_positions = self._world_positions_cache[frame]
            return self._tensor_to_bone_dict(cached_positions)
        
        # Prepare tensors for forward kinematics
        n_bones = len(self.pmx_model.bones)
        bone_offsets_tensor = torch.tensor(self.bone_offsets, dtype=torch.float32, device=self.device)
        quaternions_tensor = torch.zeros(n_bones, 4, dtype=torch.float32, device=self.device)
        local_translations_tensor = torch.zeros(n_bones, 3, dtype=torch.float32, device=self.device)
        
        # Initialize with identity quaternions
        quaternions_tensor[:, 3] = 1.0  # w component = 1
        
        # Fill in animation data for this frame
        if frame in self.vmd_data_by_frame:
            frame_data = self.vmd_data_by_frame[frame]
            
            for bone_name, data in frame_data.items():
                if bone_name in self.bone_name_to_index:
                    bone_idx = self.bone_name_to_index[bone_name]
                    
                    # Set quaternion (x,y,z,w)
                    quat = data['quaternion']
                    quaternions_tensor[bone_idx] = torch.tensor([quat[0], quat[1], quat[2], quat[3]], 
                                                               dtype=torch.float32, device=self.device)
                    
                    # Set local translation
                    pos = data['position']
                    local_translations_tensor[bone_idx] = torch.tensor([pos[0], pos[1], pos[2]], 
                                                                      dtype=torch.float32, device=self.device)
        
        # Normalize quaternions
        quaternions_tensor = quaternions_tensor / torch.norm(quaternions_tensor, dim=1, keepdim=True)
        
        # Calculate world positions using forward kinematics
        world_positions_tensor = self.fk_calculator.compute_world_positions(
            bone_offsets_tensor,
            quaternions_tensor,
            self.parent_indices,
            local_translations_tensor
        )
        
        # Cache the result
        if use_cache:
            self._world_positions_cache[frame] = world_positions_tensor.clone()
        
        return self._tensor_to_bone_dict(world_positions_tensor)
    
    def _tensor_to_bone_dict(self, world_positions_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Convert tensor results back to dictionary format."""
        world_positions_np = world_positions_tensor.cpu().numpy()
        result = {}
        
        for i, bone in enumerate(self.pmx_model.bones):
            result[bone.name] = world_positions_np[i]
        
        return result
    
    def get_bone_world_position(self, bone_name: str, frame: int) -> Optional[np.ndarray]:
        """
        Get world position of a specific bone at the specified frame.
        
        Args:
            bone_name: Name of the bone to query
            frame: Frame number to query
            
        Returns:
            World position as numpy array, or None if bone not found
        """
        if bone_name not in self.bone_name_to_index:
            return None
        
        all_positions = self.get_world_positions(frame)
        return all_positions.get(bone_name)
    
    def get_available_frames(self) -> List[int]:
        """Get list of all available frames in the motion data."""
        return sorted(list(self.vmd_data_by_frame.keys()))
    
    def get_bone_names(self) -> List[str]:
        """Get list of all bone names in the model."""
        return [bone.name for bone in self.pmx_model.bones]
    
    def get_rest_position(self, bone_name: str) -> Optional[np.ndarray]:
        """Get the rest position of a bone in MMD coordinates."""
        return self.bone_rest_positions.get(bone_name)
    
    def clear_cache(self):
        """Clear the world positions cache to free memory."""
        self._world_positions_cache.clear()


def create_animated_model(pmx_path: str, vmd_path: str, device: str = 'cpu') -> AnimatedModel:
    """
    Convenience function to create an AnimatedModel from file paths.
    
    Args:
        pmx_path: Path to PMX model file
        vmd_path: Path to VMD motion file  
        device: PyTorch device for calculations ('cpu' or 'cuda')
        
    Returns:
        AnimatedModel instance
    """
    # Load PMX model
    from ..pmx.reader import read_from_file as read_pmx
    pmx_model = read_pmx(pmx_path)
    
    # Load VMD motion
    from ..vmd.reader import read_from_file as read_vmd
    vmd_motion = read_vmd(vmd_path)
    
    return AnimatedModel(pmx_model, vmd_motion, device)