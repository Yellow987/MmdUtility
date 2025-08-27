# coding: utf-8
"""
BatchAnimatedModel class for efficiently processing multiple VMD files with a single PMX model.

Provides high-performance batch processing by loading the PMX model once and reusing it
for multiple VMD files with vectorized forward kinematics calculations.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from .. import pmx as pmx_module
from .. import vmd as vmd_module
from .batch_forward_kinematics import BatchForwardKinematics, create_bone_filter_indices


class BatchAnimatedModel:
    """
    Efficiently processes multiple VMD files using a single PMX model with batch operations.
    
    Key optimizations:
    - PMX model loaded once and reused
    - Batch forward kinematics for multiple frames
    - Optional bone filtering for performance
    - Memory-efficient chunked processing
    - Tensor reuse across VMD files
    """
    
    def __init__(self, pmx_path: str, device: str = 'cpu', chunk_size: int = 1000):
        """
        Initialize the batch animated model with a PMX file.
        
        Args:
            pmx_path: Path to PMX model file
            device: PyTorch device for calculations ('cpu' or 'cuda')
            chunk_size: Number of frames to process in each batch (memory management)
        """
        self.pmx_path = pmx_path
        self.device = device
        self.chunk_size = chunk_size
        self.batch_fk = BatchForwardKinematics(device=device)
        
        # Initialize PMX data structures
        self.pmx_model = None
        self.bone_name_to_index: Dict[str, int] = {}
        self.bone_names: List[str] = []
        self.bone_rest_positions: Dict[str, np.ndarray] = {}
        self.parent_indices: List[int] = []
        self.bone_offsets: torch.Tensor = None
        
        # Load and process PMX model once
        self._load_pmx_model()
        self._process_pmx_data()
        
        print(f"BatchAnimatedModel initialized:")
        print(f"  PMX Model: {pmx_path}")
        print(f"  Total Bones: {len(self.bone_names)}")
        print(f"  Device: {device}")
        print(f"  Chunk Size: {chunk_size} frames")
    
    def _load_pmx_model(self):
        """Load PMX model from file."""
        from ..pmx.reader import read_from_file as read_pmx
        self.pmx_model = read_pmx(self.pmx_path)
        print(f"  ✓ Loaded PMX model: {Path(self.pmx_path).name}")
    
    def _process_pmx_data(self):
        """Process PMX model data to extract bone hierarchy and rest positions."""
        if not hasattr(self.pmx_model, 'bones'):
            raise ValueError("PMX model does not contain bone data")
        
        # Build bone name to index mapping
        for i, bone in enumerate(self.pmx_model.bones):
            bone_name = bone.name
            self.bone_name_to_index[bone_name] = i
            self.bone_names.append(bone_name)
        
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
        
        # Convert to tensor for batch processing (fix performance warning)
        bone_offsets_np = np.array(bone_offsets, dtype=np.float32)
        self.bone_offsets = torch.from_numpy(bone_offsets_np).to(device=self.device)
        print(f"  ✓ Processed bone hierarchy: {n_bones} bones")
    
    def process_vmd_file(self, vmd_path: str, filter_bones: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a single VMD file and extract bone data efficiently.
        
        Args:
            vmd_path: Path to VMD motion file
            filter_bones: List of bone names to extract (None = all bones)
            
        Returns:
            Dictionary with positions, quaternions, frame_numbers, and metadata
        """
        try:
            print(f"Processing: {Path(vmd_path).name}")
            
            # Load VMD data
            from ..vmd.reader import read_from_file as read_vmd
            vmd_motion = read_vmd(vmd_path)
            
            # Process VMD data
            vmd_data_by_frame = self._process_vmd_data(vmd_motion)
            
            if not vmd_data_by_frame:
                print(f"  WARNING: No animation frames found")
                return None
            
            available_frames = sorted(list(vmd_data_by_frame.keys()))
            n_frames = len(available_frames)
            print(f"  Frames: {n_frames} (range: {min(available_frames)} to {max(available_frames)})")
            
            # Set up bone filtering
            if filter_bones is not None:
                bone_filter_indices = create_bone_filter_indices(self.bone_names, filter_bones)
                output_bone_names = filter_bones
                n_output_bones = len(filter_bones)
                print(f"  Filtering to {n_output_bones} bones: {filter_bones[:3]}...")
            else:
                bone_filter_indices = None
                output_bone_names = self.bone_names
                n_output_bones = len(self.bone_names)
            
            # Initialize output arrays
            positions = np.zeros((n_frames, n_output_bones, 3), dtype=np.float32)
            quaternions = np.zeros((n_frames, n_output_bones, 4), dtype=np.float32)
            frame_numbers = np.array(available_frames, dtype=np.int32)
            
            # Process frames in chunks for memory efficiency
            for chunk_start in range(0, n_frames, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, n_frames)
                chunk_frames = available_frames[chunk_start:chunk_end]
                chunk_size = len(chunk_frames)
                
                # Extract batch data for this chunk
                positions_chunk, quaternions_chunk = self._extract_chunk_data(
                    chunk_frames, vmd_data_by_frame, bone_filter_indices, chunk_size, n_output_bones
                )
                
                # Store results
                positions[chunk_start:chunk_end] = positions_chunk
                quaternions[chunk_start:chunk_end] = quaternions_chunk
                
                if chunk_start % (self.chunk_size * 5) == 0:  # Progress every 5 chunks
                    progress = chunk_end / n_frames * 100
                    print(f"    Progress: {progress:.1f}% ({chunk_end}/{n_frames} frames)")
            
            # Build metadata
            metadata = {
                "vmd_file": Path(vmd_path).name,
                "total_frames": n_frames,
                "bone_count": n_output_bones,
                "frame_rate": 60,
                "bone_names": output_bone_names,
                "frame_range": [int(min(available_frames)), int(max(available_frames))]
            }
            
            print(f"  ✓ Completed: {n_frames} frames processed")
            
            return {
                "positions": positions,
                "quaternions": quaternions,
                "frame_numbers": frame_numbers,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"  ERROR: Failed to process {Path(vmd_path).name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_vmd_data(self, vmd_motion) -> Dict[int, Dict[str, Dict]]:
        """Process VMD motion data and organize by frame."""
        if not hasattr(vmd_motion, 'motions'):
            raise ValueError("VMD motion does not contain bone motion data")
        
        vmd_data_by_frame = {}
        
        # Organize VMD data by frame
        for bone_frame in vmd_motion.motions:
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
                        bone_name = str(bone_name)
            
            if frame_num not in vmd_data_by_frame:
                vmd_data_by_frame[frame_num] = {}
            
            # Store position and quaternion data
            vmd_data_by_frame[frame_num][bone_name] = {
                'position': np.array([bone_frame.pos.x, bone_frame.pos.y, bone_frame.pos.z]),
                'quaternion': np.array([bone_frame.q.x, bone_frame.q.y, bone_frame.q.z, bone_frame.q.w])
            }
        
        return vmd_data_by_frame
    
    def _extract_chunk_data(self, chunk_frames: List[int], vmd_data_by_frame: Dict,
                           bone_filter_indices: Optional[List[int]], chunk_size: int, 
                           n_output_bones: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and process a chunk of frames using batch forward kinematics."""
        n_bones = len(self.bone_names)
        
        # Prepare batch tensors for forward kinematics
        quaternions_batch = torch.zeros(chunk_size, n_bones, 4, dtype=torch.float32, device=self.device)
        local_translations_batch = torch.zeros(chunk_size, n_bones, 3, dtype=torch.float32, device=self.device)
        
        # Initialize with identity quaternions
        quaternions_batch[:, :, 3] = 1.0  # w component = 1
        
        # Fill in animation data for each frame in the chunk
        for i, frame_num in enumerate(chunk_frames):
            if frame_num in vmd_data_by_frame:
                frame_data = vmd_data_by_frame[frame_num]
                
                for bone_name, data in frame_data.items():
                    if bone_name in self.bone_name_to_index:
                        bone_idx = self.bone_name_to_index[bone_name]
                        
                        # Set quaternion (x,y,z,w) - optimized tensor creation
                        quat = data['quaternion']
                        quaternions_batch[i, bone_idx, 0] = quat[0]
                        quaternions_batch[i, bone_idx, 1] = quat[1]
                        quaternions_batch[i, bone_idx, 2] = quat[2]
                        quaternions_batch[i, bone_idx, 3] = quat[3]
                        
                        # Set local translation - optimized tensor creation
                        pos = data['position']
                        local_translations_batch[i, bone_idx, 0] = pos[0]
                        local_translations_batch[i, bone_idx, 1] = pos[1]
                        local_translations_batch[i, bone_idx, 2] = pos[2]
        
        # Normalize quaternions
        quaternions_batch = quaternions_batch / torch.norm(quaternions_batch, dim=2, keepdim=True)
        
        # Compute world positions using batch forward kinematics
        world_positions_batch = self.batch_fk.compute_world_positions_batch(
            self.bone_offsets,
            quaternions_batch,
            self.parent_indices,
            local_translations_batch,
            bone_filter_indices
        )
        
        # Convert to numpy and extract quaternions for filtered bones
        positions_chunk = world_positions_batch.cpu().numpy()
        
        # Extract quaternions for filtered bones
        if bone_filter_indices is not None:
            quaternions_chunk = np.zeros((chunk_size, n_output_bones, 4), dtype=np.float32)
            for out_idx, bone_idx in enumerate(bone_filter_indices):
                quaternions_chunk[:, out_idx] = quaternions_batch[:, bone_idx].cpu().numpy()
        else:
            quaternions_chunk = quaternions_batch.cpu().numpy()
        
        return positions_chunk, quaternions_chunk
    
    def get_bone_names(self) -> List[str]:
        """Get list of all bone names in the model."""
        return self.bone_names.copy()
    
    def clear_gpu_cache(self):
        """Clear GPU cache if using CUDA."""
        if self.device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_batch_animated_model(pmx_path: str, device: str = 'cpu', chunk_size: int = 1000) -> BatchAnimatedModel:
    """
    Convenience function to create a BatchAnimatedModel.
    
    Args:
        pmx_path: Path to PMX model file
        device: PyTorch device for calculations ('cpu' or 'cuda')
        chunk_size: Number of frames to process in each batch
        
    Returns:
        BatchAnimatedModel instance
    """
    return BatchAnimatedModel(pmx_path, device, chunk_size)