# coding: utf-8
"""
Persistent PMX Model for Bone Position Calculations

Provides a persistent PMX model loader that can be initialized once and used
repeatedly for forward kinematics calculations. Generic functionality that
takes bone data and outputs world positions for specified target bones.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from ..pmx import reader as pmx_reader
from .batch_forward_kinematics import BatchForwardKinematics


class PersistentPmxModel:
    """
    Persistent PMX model for efficient bone position calculations.
    
    This class loads a PMX model once and provides methods to compute world positions
    of specified target bones given quaternion and position data for input bones.
    All operations use numpy arrays for generic compatibility.
    """
    
    def __init__(self, pmx_path: str):
        """
        Initialize with a PMX model.
        
        Args:
            pmx_path: Path to the PMX model file
        """
        self.pmx_path = Path(pmx_path)
        
        # Load PMX model (persistent)
        self.pmx_model = self._load_pmx_model()
        
        # Process PMX model data
        self.bone_name_to_index: Dict[str, int] = {}
        self.bone_index_to_name: Dict[int, str] = {}
        self.parent_indices: List[int] = []
        self.bone_offsets: np.ndarray = None
        self._process_pmx_data()
        
        print(f"✓ PersistentPmxModel initialized")
        print(f"  PMX model: {self.pmx_path}")
        print(f"  Total bones: {len(self.pmx_model.bones)}")
    
    def _load_pmx_model(self):
        """Load PMX model from file."""
        if not self.pmx_path.exists():
            raise FileNotFoundError(f"PMX model not found: {self.pmx_path}")
        
        pmx_model = pmx_reader.read_from_file(str(self.pmx_path))
        
        return pmx_model
    
    def _process_pmx_data(self):
        """Process PMX model data to extract bone hierarchy and rest positions."""
        if not hasattr(self.pmx_model, 'bones'):
            raise ValueError("PMX model does not contain bone data")
        
        # Build bone name mappings
        for i, bone in enumerate(self.pmx_model.bones):
            self.bone_name_to_index[bone.name] = i
            self.bone_index_to_name[i] = bone.name
        
        # Extract parent indices and bone offsets
        n_bones = len(self.pmx_model.bones)
        self.parent_indices = [-1] * n_bones
        bone_positions = []
        
        for i, bone in enumerate(self.pmx_model.bones):
            # Store rest position
            rest_pos = np.array([bone.position.x, bone.position.y, bone.position.z])
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
    
    def get_bone_indices(self, bone_names: List[str]) -> List[int]:
        """
        Get bone indices for given bone names.
        
        Args:
            bone_names: List of bone names
        
        Returns:
            List of bone indices (-1 for bones not found)
        """
        indices = []
        for name in bone_names:
            indices.append(self.bone_name_to_index.get(name, -1))
        return indices
    
    def compute_bone_positions(
        self,
        input_bone_names: List[str],
        quaternions: np.ndarray,
        positions: Optional[np.ndarray] = None,
        target_bone_names: Optional[List[str]] = None,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute world positions for target bones given input bone data.
        
        Args:
            input_bone_names: List of bone names that have animation data
            quaternions: [N_frames, N_input_bones, 4] quaternion data (x,y,z,w)
            positions: [N_frames, N_input_bones, 3] position data (optional, zeros if None)
            target_bone_names: List of target bone names to compute positions for
                              If None, computes for all bones
            device: Device to use for computations ('cpu' or 'cuda')
        
        Returns:
            target_positions: [N_frames, N_target_bones, 3] world positions of target bones
            target_names: List of target bone names (in order of returned positions)
        """
        n_frames = quaternions.shape[0]
        n_input_bones = len(input_bone_names)
        n_pmx_bones = len(self.pmx_model.bones)
        
        if quaternions.shape[1] != n_input_bones:
            raise ValueError(f"Quaternions shape {quaternions.shape} doesn't match input_bone_names length {n_input_bones}")
        
        if positions is None:
            positions = np.zeros((n_frames, n_input_bones, 3))
        
        # Create mappings from input bones to PMX bones
        input_to_pmx_indices = self.get_bone_indices(input_bone_names)
        
        # Create full bone arrays for all PMX bones
        full_quaternions = np.zeros((n_frames, n_pmx_bones, 4))
        full_quaternions[:, :, 3] = 1.0  # Initialize w component to 1 (identity quaternion)
        full_positions = np.zeros((n_frames, n_pmx_bones, 3))
        
        # Fill in data for bones that have animation data
        for input_idx, pmx_idx in enumerate(input_to_pmx_indices):
            if pmx_idx != -1:  # Valid bone index
                full_quaternions[:, pmx_idx] = quaternions[:, input_idx]
                full_positions[:, pmx_idx] = positions[:, input_idx]
        
        # Determine target bones
        if target_bone_names is None:
            target_bone_names = list(self.bone_name_to_index.keys())
        
        target_indices = self.get_bone_indices(target_bone_names)
        valid_targets = [(i, idx) for i, idx in enumerate(target_indices) if idx != -1]
        
        if not valid_targets:
            raise ValueError("No valid target bones found")
        
        # Set up tensors for forward kinematics
        device = torch.device(device)
        bone_offsets_tensor = torch.tensor(self.bone_offsets, dtype=torch.float32, device=device)
        quaternions_tensor = torch.tensor(full_quaternions, dtype=torch.float32, device=device)
        positions_tensor = torch.tensor(full_positions, dtype=torch.float32, device=device)
        
        # Use batch forward kinematics
        fk_calculator = BatchForwardKinematics(device=device)
        
        # Only compute positions for bones up to the maximum target index for efficiency
        max_target_idx = max(idx for _, idx in valid_targets)
        filter_indices = list(range(max_target_idx + 1))
        
        world_positions = fk_calculator.compute_world_positions_batch(
            bone_offsets=bone_offsets_tensor,
            quaternions_batch=quaternions_tensor,
            parent_indices=self.parent_indices,
            local_translations_batch=positions_tensor,
            bone_filter_indices=filter_indices
        )
        
        # Extract positions for target bones
        target_positions = np.zeros((n_frames, len(valid_targets), 3))
        actual_target_names = []
        
        for result_idx, (name_idx, pmx_idx) in enumerate(valid_targets):
            target_positions[:, result_idx] = world_positions[:, pmx_idx].cpu().numpy()
            actual_target_names.append(target_bone_names[name_idx])
        
        return target_positions, actual_target_names


def create_persistent_pmx_model(pmx_path: str) -> PersistentPmxModel:
    """
    Create a PersistentPmxModel instance.
    
    Args:
        pmx_path: Path to the PMX model file
    
    Returns:
        PersistentPmxModel instance
    """
    return PersistentPmxModel(pmx_path)