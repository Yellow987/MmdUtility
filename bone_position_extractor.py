"""
Bone Position Extractor Module

This module provides functionality to extract bone positions for every frame
from PMX model files and VMD motion files.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
try:
    # Try relative import first (when used as part of the package)
    from .pymeshio.pmx import reader as pmx_reader
    from .pymeshio.vmd import reader as vmd_reader
except ImportError:
    # Fallback to absolute import (when run directly)
    from pymeshio.pmx import reader as pmx_reader
    from pymeshio.vmd import reader as vmd_reader

# Pytransform3d integration
try:
    from pytransform3d.transformations import TransformManager
    HAS_PYTRANSFORM3D = True
except ImportError:
    HAS_PYTRANSFORM3D = False
    TransformManager = None


class BonePositionExtractor:
    """
    Extracts bone positions for every frame given a PMX model and VMD motion.
    
    Usage:
        extractor = BonePositionExtractor()
        positions = extractor.extract_bone_positions(pmx_path, vmd_path)
    """
    
    def __init__(self):
        self.pmx_model = None
        self.vmd_motion = None
        self.bone_hierarchy = {}
        self.bone_name_to_index = {}
        self.transform_manager = None
        
    def load_pmx(self, pmx_path: str):
        """
        Load PMX model file.
        
        Args:
            pmx_path: Path to the PMX file
        """
        try:
            self.pmx_model = pmx_reader.read_from_file(pmx_path)
            if self.pmx_model is None:
                raise ValueError(f"Failed to load PMX file: {pmx_path}")
            
            # Build bone hierarchy and name mapping
            self._build_bone_hierarchy()
            print(f"Successfully loaded PMX: {self.pmx_model.name}")
            print(f"Bones loaded: {len(self.pmx_model.bones)}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading PMX file {pmx_path}: {str(e)}")
    
    def load_vmd(self, vmd_path: str):
        """
        Load VMD motion file.
        
        Args:
            vmd_path: Path to the VMD file
        """
        try:
            self.vmd_motion = vmd_reader.read_from_file(vmd_path)
            if self.vmd_motion is None:
                raise ValueError(f"Failed to load VMD file: {vmd_path}")
                
            print(f"Successfully loaded VMD: {self.vmd_motion.model_name}")
            print(f"Bone motions loaded: {len(self.vmd_motion.motions)}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading VMD file {vmd_path}: {str(e)}")
    
    def _build_bone_hierarchy(self):
        """
        Build bone hierarchy from PMX model for efficient lookups.
        """
        if not self.pmx_model or not self.pmx_model.bones:
            return
            
        self.bone_hierarchy = {}
        self.bone_name_to_index = {}
        
        for i, bone in enumerate(self.pmx_model.bones):
            # Map bone name to index
            bone_name = bone.name
            if isinstance(bone_name, bytes):
                bone_name = bone_name.decode('utf-8', errors='ignore')
            
            self.bone_name_to_index[bone_name] = i
            
            # Store bone hierarchy info
            self.bone_hierarchy[i] = {
                'name': bone_name,
                'parent_index': bone.parent_index,
                'position': (bone.position.x, bone.position.y, bone.position.z),
                'children': []
            }
        
        # Build parent-child relationships
        for i, bone_info in self.bone_hierarchy.items():
            parent_idx = bone_info['parent_index']
            if parent_idx != -1 and parent_idx in self.bone_hierarchy:
                self.bone_hierarchy[parent_idx]['children'].append(i)
    
    def get_rest_pose_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get the world positions of all bones in rest pose (T-pose).
        
        Returns:
            Dictionary mapping bone names to their (x, y, z) world positions
        """
        if not self.pmx_model or not self.bone_hierarchy:
            raise ValueError("PMX model must be loaded first")
        
        # Calculate world positions for all bones
        world_positions = {}
        
        def calculate_world_position(bone_index: int, parent_world_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
            """
            Recursively calculate world position for a bone and its children.
            """
            if bone_index not in self.bone_hierarchy:
                return parent_world_pos
            
            bone_info = self.bone_hierarchy[bone_index]
            local_pos = bone_info['position']
            
            # Calculate world position by adding local position to parent's world position
            world_pos = (
                parent_world_pos[0] + local_pos[0],
                parent_world_pos[1] + local_pos[1],
                parent_world_pos[2] + local_pos[2]
            )
            
            # Store the world position, handling duplicate names
            bone_name = bone_info['name']
            if bone_name in world_positions:
                # Handle duplicate bone names by appending index
                unique_name = f"{bone_name}_{bone_index}"
                world_positions[unique_name] = world_pos
            else:
                world_positions[bone_name] = world_pos
            
            # Process children
            for child_index in bone_info['children']:
                calculate_world_position(child_index, world_pos)
            
            return world_pos
        
        # Start with root bones (bones with no parent, parent_index = -1)
        root_bones = [i for i, info in self.bone_hierarchy.items() if info['parent_index'] == -1]
        
        for root_bone_index in root_bones:
            calculate_world_position(root_bone_index)
        
        # Handle any orphaned bones that weren't processed (bones with invalid parent references)
        processed_bones = set(world_positions.keys())
        all_bone_names = {info['name'] for info in self.bone_hierarchy.values()}
        orphaned_bones = all_bone_names - processed_bones
        
        if orphaned_bones:
            print(f"Debug: Found {len(orphaned_bones)} unprocessed bones: {list(orphaned_bones)}")
            # Process orphaned bones as if they were root bones
            for bone_index, bone_info in self.bone_hierarchy.items():
                if bone_info['name'] in orphaned_bones:
                    calculate_world_position(bone_index)
        
        return world_positions
    
    def _initialize_transform_manager(self):
        """
        Initialize TransformManager for skeleton transformations.
        """
        if not HAS_PYTRANSFORM3D:
            raise ImportError(
                "pytransform3d is required for skeleton transforms. "
                "Install with: pip install pytransform3d"
            )
        self.transform_manager = TransformManager()
    
    def _load_skeleton_to_transform_manager(self):
        """
        Load PMX skeleton into TransformManager with rest pose transforms.
        
        This creates a coordinate frame for each bone with the proper parent-child
        relationships and rest pose positions.
        """
        if not self.pmx_model or not self.bone_hierarchy:
            raise ValueError("PMX model must be loaded first")
        
        self._initialize_transform_manager()
        
        # Add world coordinate frame
        self.transform_manager.add_transform("world", "world", np.eye(4))
        
        # Keep track of processed bones to handle dependencies
        processed_bones = set()
        
        def process_bone(bone_index):
            """Recursively process bone and its dependencies."""
            if bone_index in processed_bones:
                return
                
            bone_info = self.bone_hierarchy[bone_index]
            bone_name = f"bone_{bone_index}"  # Unique frame name
            parent_idx = bone_info['parent_index']
            
            # Ensure parent is processed first
            if parent_idx != -1 and parent_idx not in processed_bones:
                process_bone(parent_idx)
            
            # Create transformation matrix for this bone
            local_pos = bone_info['position']
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = local_pos  # Set translation (rest pose position)
            
            # Add transform to manager
            parent_frame = "world" if parent_idx == -1 else f"bone_{parent_idx}"
            self.transform_manager.add_transform(
                parent_frame, bone_name, transform_matrix
            )
            
            processed_bones.add(bone_index)
        
        # Process all bones (parent dependencies handled automatically)
        for bone_index in self.bone_hierarchy:
            process_bone(bone_index)
        
        print(f"Loaded {len(processed_bones)} bones into TransformManager")
    
    def get_rest_pose_positions_pytransform3d(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get rest pose positions using TransformManager.
        
        Returns:
            Dictionary mapping bone names to their (x, y, z) world positions
        """
        if not self.transform_manager:
            self._load_skeleton_to_transform_manager()
        
        positions = {}
        
        for bone_index, bone_info in self.bone_hierarchy.items():
            bone_name = bone_info['name']
            frame_name = f"bone_{bone_index}"
            
            try:
                # Get world transform for this bone
                world_transform = self.transform_manager.get_transform(
                    "world", frame_name
                )
                
                # Extract position (translation part of transformation matrix)
                world_pos = world_transform[:3, 3]
                positions[bone_name] = tuple(world_pos)
                
            except Exception as e:
                print(f"Warning: Could not get transform for bone {bone_name}: {e}")
                # Fallback to original calculation
                positions[bone_name] = (0.0, 0.0, 0.0)
        
        return positions
    
    def get_rest_pose_positions_array(self) -> np.ndarray:
        """
        Get rest pose positions as a numpy array.
        
        Returns:
            numpy array of shape (num_bones, 3) with XYZ positions
        """
        positions_dict = self.get_rest_pose_positions()
        
        # Create array in bone index order
        positions_array = np.zeros((len(self.bone_hierarchy), 3))
        
        for bone_index, bone_info in self.bone_hierarchy.items():
            bone_name = bone_info['name']
            if bone_name in positions_dict:
                positions_array[bone_index] = positions_dict[bone_name]
        
        return positions_array
    
    def extract_bone_positions(self, pmx_path: str, vmd_path: str) -> List[Dict[str, Tuple[float, float, float]]]:
        """
        Extract bone positions for every frame.
        
        Args:
            pmx_path: Path to the PMX model file
            vmd_path: Path to the VMD motion file
            
        Returns:
            List of dictionaries, where each dictionary maps bone names to their
            (x, y, z) world positions for that frame
        """
        # Load files
        self.load_pmx(pmx_path)
        self.load_vmd(vmd_path)
        
        if not self.pmx_model or not self.vmd_motion:
            raise ValueError("Both PMX model and VMD motion must be loaded")
        
        # TODO: Implement actual bone position calculation
        # For now, return empty structure showing the intended format
        print("Note: Bone position calculation not yet implemented in this draft")
        print("This module currently only loads the PMX and VMD files")
        
        # Placeholder return structure
        frame_positions = []
        
        # Show what the final structure will look like
        sample_frame = {}
        for bone_name in self.bone_name_to_index.keys():
            sample_frame[bone_name] = (0.0, 0.0, 0.0)  # Placeholder positions
        
        frame_positions.append(sample_frame)
        
        return frame_positions
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.pmx_model:
            return {"error": "No PMX model loaded"}
        
        return {
            "model_name": self.pmx_model.name,
            "english_name": getattr(self.pmx_model, 'english_name', ''),
            "bone_count": len(self.pmx_model.bones),
            "vertex_count": len(self.pmx_model.vertices),
            "material_count": len(self.pmx_model.materials)
        }
    
    def get_motion_info(self) -> Dict:
        """
        Get information about the loaded motion.
        
        Returns:
            Dictionary containing motion information
        """
        if not self.vmd_motion:
            return {"error": "No VMD motion loaded"}
        
        # Find frame range
        max_frame = 0
        if self.vmd_motion.motions:
            max_frame = max(motion.frame for motion in self.vmd_motion.motions)
        
        return {
            "model_name": self.vmd_motion.model_name,
            "bone_motion_count": len(self.vmd_motion.motions),
            "morph_count": len(self.vmd_motion.shapes),
            "camera_frame_count": len(self.vmd_motion.cameras),
            "max_frame": max_frame
        }


# Convenience functions for direct usage
def extract_bone_positions(pmx_path: str, vmd_path: str) -> List[Dict[str, Tuple[float, float, float]]]:
    """
    Convenience function to extract bone positions directly.
    
    Args:
        pmx_path: Path to the PMX model file
        vmd_path: Path to the VMD motion file
        
    Returns:
        List of dictionaries mapping bone names to (x, y, z) positions per frame
    """
    extractor = BonePositionExtractor()
    return extractor.extract_bone_positions(pmx_path, vmd_path)


def get_rest_pose_positions(pmx_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Convenience function to get rest pose positions directly.
    
    Args:
        pmx_path: Path to the PMX model file
        
    Returns:
        Dictionary mapping bone names to (x, y, z) rest positions
    """
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    return extractor.get_rest_pose_positions()


def get_rest_pose_positions_array(pmx_path: str) -> np.ndarray:
    """
    Convenience function to get rest pose positions as numpy array.
    
    Args:
        pmx_path: Path to the PMX model file
        
    Returns:
        numpy array of shape (num_bones, 3) with XYZ positions
    """
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    return extractor.get_rest_pose_positions_array()


def get_file_info(pmx_path: str, vmd_path: str) -> Tuple[Dict, Dict]:
    """
    Get information about PMX and VMD files.
    
    Args:
        pmx_path: Path to the PMX model file
        vmd_path: Path to the VMD motion file
        
    Returns:
        Tuple of (model_info, motion_info) dictionaries
    """
    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)
    extractor.load_vmd(vmd_path)
    return extractor.get_model_info(), extractor.get_motion_info()