#!/usr/bin/env python3
"""
Skeleton Quaternion Validation Script

This script loads a PMX model, extracts quaternions from JSON animation data,
performs forward kinematics using PyTorch, and validates the derived positions
against the JSON positions for frame 0.

Requirements:
- PyTorch
- NumPy
- MmdUtility (pymeshio)

Usage:
    python validate_skeleton_quaternions.py
"""

import torch
import numpy as np
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import pymeshio directly since we're in the MmdUtility directory
try:
    from pymeshio.pmx import reader as pmx_reader
    from pymeshio.vmd import reader as vmd_reader
    print("✓ Successfully imported pymeshio.pmx.reader and pymeshio.vmd.reader")
except ImportError as e:
    print(f"✗ Failed to import pymeshio: {e}")
    sys.exit(1)


class PMXLoader:
    """Handles PMX model loading and bone hierarchy extraction."""
    
    def __init__(self):
        self.model = None
        self.bone_hierarchy = {}
        self.bone_name_to_index = {}
        self.bone_index_to_name = {}
    
    def load_pmx(self, pmx_path: str) -> bool:
        """Load PMX model and build bone hierarchy."""
        try:
            self.model = pmx_reader.read_from_file(pmx_path)
            if not self.model:
                print(f"✗ Failed to load PMX model: {pmx_path}")
                return False
            
            print(f"✓ Loaded PMX model: {self.model.name}")
            print(f"  Total bones: {len(self.model.bones)}")
            
            # Build bone mappings and hierarchy
            self._build_bone_hierarchy()
            return True
            
        except Exception as e:
            print(f"✗ Error loading PMX file: {e}")
            return False
    
    def _build_bone_hierarchy(self):
        """Build bone hierarchy and name mappings."""
        # First pass: store world positions and build mappings
        bind_world_positions = {}
        
        for i, bone in enumerate(self.model.bones):
            bone_name = bone.name.strip()
            
            # Apply coordinate conversion (Left-handed Y-up to Right-handed Z-up)
            world_pos = self._convert_coord(bone.position)
            bind_world_positions[i] = world_pos
            
            self.bone_hierarchy[i] = {
                'name': bone_name,
                'english_name': getattr(bone, 'english_name', '').strip(),
                'world_position': world_pos,
                'parent_index': bone.parent_index,
                'index': i
            }
            
            self.bone_name_to_index[bone_name] = i
            self.bone_index_to_name[i] = bone_name
            
            # Also map English name if available
            if hasattr(bone, 'english_name') and bone.english_name.strip():
                english_name = bone.english_name.strip()
                self.bone_name_to_index[english_name] = i
        
        # Second pass: calculate bone offsets from parent
        for i, bone_info in self.bone_hierarchy.items():
            parent_idx = bone_info['parent_index']
            
            if parent_idx == -1:  # Root bone
                bone_info['offset'] = (0.0, 0.0, 0.0)  # Root offset is zero
            else:  # Child bone
                parent_world_pos = bind_world_positions[parent_idx]
                child_world_pos = bone_info['world_position']
                # Offset = child_world - parent_world
                bone_info['offset'] = tuple(np.array(child_world_pos) - np.array(parent_world_pos))
    
    def _convert_coord(self, pos, scale=1.0):
        """Convert Left-handed Y-up to Right-handed Z-up coordinate system."""
        return (pos.x * scale, pos.z * scale, pos.y * scale)
    
    def get_bone_info(self, bone_identifier) -> Optional[Dict]:
        """Get bone info by name or index."""
        if isinstance(bone_identifier, str):
            if bone_identifier in self.bone_name_to_index:
                index = self.bone_name_to_index[bone_identifier]
                return self.bone_hierarchy[index]
        elif isinstance(bone_identifier, int):
            return self.bone_hierarchy.get(bone_identifier)
        return None


class QuaternionParser:
    """Handles JSON quaternion data parsing."""
    
    def __init__(self):
        self.frame_data = None
        self.bone_data = {}
    
    def load_json(self, json_path: str, target_frame: int = 0) -> bool:
        """Load JSON data and extract target frame."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✓ Loaded JSON file: {json_path}")
            print(f"  Total frames: {data['metadata']['totalFrames']}")
            print(f"  Core bones: {data['metadata']['coreBoneCount']}")
            
            # Extract target frame
            if target_frame < len(data['frames']):
                self.frame_data = data['frames'][target_frame]
                print(f"✓ Extracted frame {target_frame} data")
                print(f"  Bones in frame: {len(self.frame_data['bones'])}")
                
                self._parse_bone_data()
                return True
            else:
                print(f"✗ Frame {target_frame} not found in JSON")
                return False
                
        except Exception as e:
            print(f"✗ Error loading JSON file: {e}")
            return False
    
    def _parse_bone_data(self):
        """Parse bone data from frame."""
        for bone_name, bone_info in self.frame_data['bones'].items():
            pos = bone_info['position']
            quat = bone_info['quaternion']
            
            # Convert JSON coordinates same as PMX: (x,y,z) -> (x,z,y) for LH Y-up to RH Z-up
            position = self._convert_json_position(pos)
            
            # Normalize quaternion to avoid numerical drift
            quaternion = np.array([quat['x'], quat['y'], quat['z'], quat['w']])
            quaternion = quaternion / np.linalg.norm(quaternion)  # Normalize
            
            self.bone_data[bone_name] = {
                'position': position,
                'quaternion': quaternion,
                'bone_index': bone_info['boneIndex']
            }
    
    def _convert_json_position(self, pos):
        """Convert JSON position from MMD LH Y-up to RH Z-up coordinate system."""
        # Same conversion as PMX: (x,y,z) -> (x,z,y)
        return np.array([pos['x'], pos['z'], pos['y']], dtype=np.float32)
    
    def get_bone_quaternion(self, bone_name: str) -> Optional[np.ndarray]:
        """Get quaternion for a bone."""
        return self.bone_data.get(bone_name, {}).get('quaternion')
    
    def get_bone_position(self, bone_name: str) -> Optional[np.ndarray]:
        """Get world position for a bone."""
        return self.bone_data.get(bone_name, {}).get('position')


class VMDLoader:
    """Handles VMD motion data loading."""
    
    def __init__(self):
        self.vmd_model = None
        self.bone_motions = {}
    
    def load_vmd(self, vmd_path: str, target_frame: int = 0) -> bool:
        """Load VMD motion data and extract target frame."""
        try:
            self.vmd_model = vmd_reader.read_from_file(vmd_path)
            if not self.vmd_model:
                print(f"✗ Failed to load VMD file: {vmd_path}")
                return False
            
            print(f"✓ Loaded VMD file: {vmd_path}")
            
            # Check what attributes are available
            if hasattr(self.vmd_model, 'bone_list'):
                bone_motions = self.vmd_model.bone_list
            elif hasattr(self.vmd_model, 'bones'):
                bone_motions = self.vmd_model.bones
            elif hasattr(self.vmd_model, 'motions'):
                bone_motions = self.vmd_model.motions
            else:
                # Print available attributes to debug
                attrs = [attr for attr in dir(self.vmd_model) if not attr.startswith('_')]
                print(f"  Available VMD attributes: {attrs}")
                print(f"✗ Could not find bone motions in VMD model")
                return False
            
            print(f"  Total bone motions: {len(bone_motions)}")
            
            # Extract bone motions for target frame
            self._extract_frame_motions(target_frame, bone_motions)
            return True
            
        except Exception as e:
            print(f"✗ Error loading VMD file: {e}")
            return False
    
    def _extract_frame_motions(self, target_frame: int, bone_motions):
        """Extract bone motions for target frame."""
        for bone_motion in bone_motions:
            # Handle bone name encoding (VMD names might be bytes)
            bone_name_raw = bone_motion.name
            if isinstance(bone_name_raw, bytes):
                try:
                    bone_name = bone_name_raw.decode('shift_jis').strip()
                except UnicodeDecodeError:
                    try:
                        bone_name = bone_name_raw.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        bone_name = bone_name_raw.decode('cp932', errors='ignore').strip()
            else:
                bone_name = str(bone_name_raw).strip()
            
            if bone_motion.frame == target_frame:
                # Check available attributes on first bone_motion for debugging
                if len(self.bone_motions) == 0:
                    attrs = [attr for attr in dir(bone_motion) if not attr.startswith('_')]
                    print(f"    BoneFrame attributes: {attrs}")
                
                # Try different attribute names for position
                pos = None
                if hasattr(bone_motion, 'position'):
                    pos = bone_motion.position
                elif hasattr(bone_motion, 'translation'):
                    pos = bone_motion.translation
                elif hasattr(bone_motion, 'pos'):
                    pos = bone_motion.pos
                elif hasattr(bone_motion, 'location'):
                    pos = bone_motion.location
                else:
                    print(f"    Warning: Could not find position attribute for bone {bone_name}")
                    pos = type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0})()
                
                # Try different attribute names for rotation
                rot = None
                if hasattr(bone_motion, 'q'):
                    rot = bone_motion.q
                elif hasattr(bone_motion, 'rotation'):
                    rot = bone_motion.rotation
                elif hasattr(bone_motion, 'quaternion'):
                    rot = bone_motion.quaternion
                elif hasattr(bone_motion, 'rot'):
                    rot = bone_motion.rot
                else:
                    print(f"    Warning: Could not find rotation attribute for bone {bone_name}")
                    rot = type('obj', (object,), {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0})()
                
                # Apply coordinate conversion for position (Y-up LH to Z-up RH)
                converted_pos = self._convert_coord(pos)
                
                # Apply coordinate conversion for rotation if needed
                converted_rot = np.array([rot.x, rot.y, rot.z, rot.w])
                # Normalize quaternion
                converted_rot = converted_rot / np.linalg.norm(converted_rot)
                
                self.bone_motions[bone_name] = {
                    'position': np.array(converted_pos),
                    'quaternion': converted_rot,
                    'frame': target_frame
                }
    
    def _convert_coord(self, pos, scale=1.0):
        """Convert Left-handed Y-up to Right-handed Z-up coordinate system."""
        return (pos.x * scale, pos.z * scale, pos.y * scale)
    
    def get_bone_translation(self, bone_name: str) -> Optional[np.ndarray]:
        """Get local translation for a bone at the target frame."""
        motion = self.bone_motions.get(bone_name)
        return motion['position'] if motion else None
    
    def get_bone_rotation(self, bone_name: str) -> Optional[np.ndarray]:
        """Get local rotation for a bone at the target frame."""
        motion = self.bone_motions.get(bone_name)
        return motion['quaternion'] if motion else None


class ForwardKinematics:
    """PyTorch-based forward kinematics engine."""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
    def compute_world_positions(self,
                              bone_offsets: torch.Tensor,
                              quaternions: torch.Tensor,
                              local_translations: torch.Tensor,
                              parent_indices: List[int],
                              root_translation: torch.Tensor = None,
                              root_index: int = None) -> torch.Tensor:
        """
        Compute world positions using forward kinematics.
        
        Args:
            bone_offsets: [N, 3] tensor of bone offsets from parent (bind pose)
            quaternions: [N, 4] tensor of local rotations (x,y,z,w) - normalized
            local_translations: [N, 3] tensor of per-frame local translations
            parent_indices: List of parent bone indices (-1 for root)
            root_translation: [3] tensor for root bone translation override
            root_index: Index of the specific root bone to apply root_translation to
            
        Returns:
            world_positions: [N, 3] tensor of world positions
        """
        n_bones = bone_offsets.shape[0]
        world_positions = torch.zeros_like(bone_offsets)
        world_rotations = quaternions.clone()  # Start with normalized quaternions
        
        # Pre-allocate tensors to avoid creating new ones in loop
        zero_vec = torch.zeros(3, device=self.device, dtype=bone_offsets.dtype)
        identity_quat = torch.tensor([0, 0, 0, 1], device=self.device, dtype=quaternions.dtype)
        
        # Get traversal order (parents before children)
        traversal_order = self._get_traversal_order(parent_indices)
        
        for j in traversal_order:
            p = parent_indices[j]
            
            if p == -1:  # Root bone
                if (root_translation is not None) and (root_index is not None) and (j == root_index):
                    world_positions[j] = root_translation + local_translations[j]
                else:
                    world_positions[j] = local_translations[j]  # root offset is zero
                world_rotations[j] = quaternions[j]
            else:  # Child bone
                parent_rot = world_rotations[p]
                world_rotations[j] = self._quaternion_multiply(parent_rot, quaternions[j])
                step = bone_offsets[j] + local_translations[j]  # assuming parent-space locals
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
        """Multiply two quaternions: q1 * q2. More efficient version."""
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        
        # Reuse existing tensor instead of creating new one
        result = torch.empty_like(q1)
        result[0] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        result[1] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        result[2] = w1*z2 + x1*y2 - y1*x2 + z1*w2
        result[3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return result
    
    def _rotate_vector_by_quaternion(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q using efficient method."""
        # More efficient quaternion-vector rotation: v' = q * [v, 0] * q*
        # Using the formula: v' = v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)
        q_vec = q[:3]  # x, y, z components
        q_w = q[3]     # w component
        
        # First cross product: cross(q_vec, v) + q_w * v
        cross1 = torch.linalg.cross(q_vec, v) + q_w * v
        
        # Second cross product: cross(q_vec, cross1)
        cross2 = torch.linalg.cross(q_vec, cross1)
        
        # Final result: v + 2 * cross2
        return v + 2.0 * cross2


class ValidationEngine:
    """Handles position validation and error metrics."""
    
    def __init__(self, tolerance: float = 0.001):
        self.tolerance = tolerance
        self.results = {}
    
    def validate_positions(self, 
                         derived_positions: Dict[str, np.ndarray],
                         json_positions: Dict[str, np.ndarray]) -> Dict:
        """
        Validate derived positions against JSON positions.
        
        Returns:
            validation_results: Dict with per-bone errors and summary statistics
        """
        bone_errors = {}
        errors = []
        failed_bones = []
        
        for bone_name in derived_positions.keys():
            if bone_name in json_positions:
                derived_pos = derived_positions[bone_name]
                json_pos = json_positions[bone_name]
                
                error = np.linalg.norm(derived_pos - json_pos)
                bone_errors[bone_name] = error
                errors.append(error)
                
                if error > self.tolerance:
                    failed_bones.append((bone_name, error))
        
        # Calculate summary statistics
        errors = np.array(errors)
        summary = {
            'total_bones': len(bone_errors),
            'mean_error': np.mean(errors) if len(errors) > 0 else 0,
            'max_error': np.max(errors) if len(errors) > 0 else 0,
            'min_error': np.min(errors) if len(errors) > 0 else 0,
            'within_tolerance': len(errors) - len(failed_bones),
            'tolerance_percentage': ((len(errors) - len(failed_bones)) / len(errors) * 100) if len(errors) > 0 else 0,
            'failed_bones': failed_bones
        }
        
        return {
            'bone_errors': bone_errors,
            'summary': summary,
            'tolerance': self.tolerance,
            'passed': len(failed_bones) == 0
        }
    
    def generate_report(self, results: Dict, pmx_path: str, json_path: str, frame: int):
        """Generate comprehensive validation report."""
        print("\n" + "="*60)
        print("SKELETON QUATERNION VALIDATION REPORT")
        print("="*60)
        print(f"PMX File: {pmx_path}")
        print(f"JSON File: {json_path}")
        print(f"Frame: {frame} (Rest Pose)")
        print(f"Tolerance: {results['tolerance']} units")
        
        summary = results['summary']
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Total Bones Validated: {summary['total_bones']}")
        print(f"Mean Position Error: {summary['mean_error']:.6f} units")
        print(f"Max Position Error: {summary['max_error']:.6f} units")
        print(f"Min Position Error: {summary['min_error']:.6f} units")
        print(f"Bones Within Tolerance: {summary['within_tolerance']}/{summary['total_bones']} ({summary['tolerance_percentage']:.1f}%)")
        
        print(f"\n=== VALIDATION RESULT ===")
        if results['passed']:
            print("✅ PASSED - All bones within tolerance threshold")
        else:
            print("❌ FAILED - Some bones exceed tolerance threshold")
            print(f"\nFailed bones ({len(summary['failed_bones'])}):")
            for bone_name, error in summary['failed_bones']:
                print(f"  {bone_name}: Error={error:.6f} units")
        
        # Show top 10 bones by error
        bone_errors = results['bone_errors']
        sorted_bones = sorted(bone_errors.items(), key=lambda x: x[1], reverse=True)
        print(f"\n=== TOP 10 BONES BY ERROR ===")
        for i, (bone_name, error) in enumerate(sorted_bones[:10]):
            status = "PASS" if error <= results['tolerance'] else "FAIL"
            print(f"{i+1:2d}. {bone_name}: {error:.6f} units ({status})")


def main():
    """Main validation function."""
    print("Skeleton Quaternion Validation Script")
    print("====================================")
    
    # File paths (relative to MmdUtility directory)
    pmx_path = "test/pdtt.pmx"
    json_path = "test/dan_alivef_01.imo_bone_positions_quaternions.json"
    vmd_path = "test/dan_alivef_01.imo.vmd"
    target_frame = 0
    tolerance = 0.001
    
    # Validate file existence
    if not os.path.exists(pmx_path):
        print(f"✗ PMX file not found: {pmx_path}")
        return False
    
    if not os.path.exists(json_path):
        print(f"✗ JSON file not found: {json_path}")
        return False
        
    if not os.path.exists(vmd_path):
        print(f"✗ VMD file not found: {vmd_path}")
        return False
    
    print(f"Using PMX file: {pmx_path}")
    print(f"Using JSON file: {json_path}")
    print(f"Using VMD file: {vmd_path}")
    print(f"Target frame: {target_frame}")
    print(f"Tolerance: {tolerance} units")
    print()
    
    # Initialize components
    pmx_loader = PMXLoader()
    json_parser = QuaternionParser()
    vmd_loader = VMDLoader()
    fk_engine = ForwardKinematics()
    validator = ValidationEngine(tolerance)
    
    # Load PMX model
    print("Step 1: Loading PMX model...")
    if not pmx_loader.load_pmx(pmx_path):
        return False
    
    # Load JSON data
    print("\nStep 2: Loading JSON data...")
    if not json_parser.load_json(json_path, target_frame):
        return False
    
    # Load VMD data
    print("\nStep 3: Loading VMD data...")
    if not vmd_loader.load_vmd(vmd_path, target_frame):
        return False
    
    # Extract skeleton data
    print("\nStep 4: Processing bone data...")
    
    # Build compact list and remap parent indices
    selected = []
    for bone_name in json_parser.bone_data.keys():
        bone_info = pmx_loader.get_bone_info(bone_name)
        if bone_info is not None:
            selected.append(bone_info['index'])
    
    # Create old -> new index mapping
    old2new = {old_i: new_i for new_i, old_i in enumerate(selected)}
    
    print(f"✓ Found {len(selected)} matching bones")
    print(f"  Selected bone indices: {sorted(selected)}")
    
    # Reindex parents; if parent not in selected, set -1
    parent_indices = []
    bone_offsets = []
    quaternions = []
    local_translations = []
    bone_names = []
    json_positions = {}
    
    for old_i in selected:
        bone_info = pmx_loader.get_bone_info(old_i)
        bone_names.append(bone_info['name'])
        bone_offsets.append(bone_info['offset'])
        
        # Get quaternion from JSON (already normalized)
        quat = json_parser.get_bone_quaternion(bone_info['name'])
        quaternions.append(quat)
        
        # Get local translation from VMD for specific bones
        bone_name = bone_info['name']
        vmd_translation = vmd_loader.get_bone_translation(bone_name)
        
        if bone_name in ['センター', 'Center']:
            # Center bone: uses XZ positioning from VMD
            if vmd_translation is not None:
                # Apply XZ from VMD, keep Y=0
                local_translations.append([vmd_translation[0], 0.0, vmd_translation[2]])
                print(f"    Center translation (XZ): [{vmd_translation[0]:.6f}, 0.0, {vmd_translation[2]:.6f}]")
            else:
                local_translations.append([0.0, 0.0, 0.0])
                
        elif bone_name in ['グルーブ', 'Groove']:
            # Groove bone: uses Y positioning from VMD
            if vmd_translation is not None:
                # Apply Y from VMD, keep XZ=0
                local_translations.append([0.0, vmd_translation[1], 0.0])
                print(f"    Groove translation (Y): [0.0, {vmd_translation[1]:.6f}, 0.0]")
            else:
                local_translations.append([0.0, 0.0, 0.0])
        else:
            # Other bones: no local translation
            local_translations.append([0.0, 0.0, 0.0])
        
        # Remap parent index to compact array
        p_old = bone_info['parent_index']
        parent_indices.append(old2new.get(p_old, -1))
        
        # Store JSON world position for validation
        json_positions[bone_info['name']] = json_parser.get_bone_position(bone_info['name'])
    
    # Find Center's compact index and calculate root translation
    center_new_idx = None
    root_translation_tensor = None
    
    for name in ['センター', 'Center']:
        if name in pmx_loader.bone_name_to_index:
            old = pmx_loader.bone_name_to_index[name]
            if old in old2new:
                center_new_idx = old2new[old]
                # Calculate root translation (relative to bind pose)
                center_bind = torch.tensor(pmx_loader.bone_hierarchy[old]['world_position'], dtype=torch.float32)
                center_json = torch.tensor(json_positions[name], dtype=torch.float32)
                root_translation_tensor = center_json - center_bind
                print(f"  Center bone found: {name}")
                print(f"    Old index: {old}, New index: {center_new_idx}")
                print(f"    Bind position: {center_bind}")
                print(f"    JSON position: {center_json}")
                print(f"    Root translation: {root_translation_tensor}")
                break
    
    # Convert to tensors with proper normalization
    bone_offsets_tensor = torch.tensor(np.array(bone_offsets), dtype=torch.float32)
    quaternions_tensor = torch.tensor(np.array(quaternions), dtype=torch.float32)
    local_translations_tensor = torch.tensor(np.array(local_translations), dtype=torch.float32)
    
    # Normalize quaternions to handle numerical drift
    quaternions_tensor = quaternions_tensor / torch.norm(quaternions_tensor, dim=1, keepdim=True)
    
    print(f"  Bone offsets shape: {bone_offsets_tensor.shape}")
    print(f"  Quaternions shape: {quaternions_tensor.shape}")
    print(f"  Local translations shape: {local_translations_tensor.shape}")
    print(f"  Parent indices (remapped): {parent_indices}")
    
    # Debugging: Temporarily zero out local translations for sanity check
    print("\n--- DEBUGGING: Testing with zero local translations ---")
    local_translations_zero = local_translations_tensor.clone()
    local_translations_zero.zero_()
    
    # Perform forward kinematics with zeroed local translations
    print("\nStep 5a: Computing FK with zero local translations...")
    derived_positions_zero = fk_engine.compute_world_positions(
        bone_offsets_tensor,
        quaternions_tensor,
        local_translations_zero,
        parent_indices,
        root_translation=root_translation_tensor,
        root_index=center_new_idx
    )
    
    # Convert to dictionary for validation
    derived_positions_zero_dict = {}
    for i, bone_name in enumerate(bone_names):
        derived_positions_zero_dict[bone_name] = derived_positions_zero[i].numpy()
    
    # Validate zero local translations
    results_zero = validator.validate_positions(derived_positions_zero_dict, json_positions)
    print(f"Mean error with zero local translations: {results_zero['summary']['mean_error']:.6f}")
    
    # Perform forward kinematics with original local translations
    print("\nStep 5b: Computing FK with original local translations...")
    derived_positions_tensor = fk_engine.compute_world_positions(
        bone_offsets_tensor,
        quaternions_tensor,
        local_translations_tensor,
        parent_indices,
        root_translation=root_translation_tensor,
        root_index=center_new_idx
    )
    
    # Convert back to dictionary
    derived_positions = {}
    for i, bone_name in enumerate(bone_names):
        derived_positions[bone_name] = derived_positions_tensor[i].numpy()
    
    print(f"✓ Computed world positions for {len(derived_positions)} bones")
    
    # Pelvis-relative validation (quick structural check)
    print("\n--- DEBUGGING: Pelvis-relative validation ---")
    pelvis = '下半身' if '下半身' in bone_names else 'LowerBody'
    if pelvis in derived_positions and pelvis in json_positions:
        # Make copies for pelvis-relative comparison
        derived_rel = {}
        json_rel = {}
        p0 = derived_positions[pelvis].copy()
        j0 = json_positions[pelvis].copy()
        
        for k in derived_positions:
            derived_rel[k] = derived_positions[k] - p0
            json_rel[k] = json_positions[k] - j0
        
        # Validate pelvis-relative positions
        results_pelvis = validator.validate_positions(derived_rel, json_rel)
        print(f"Pelvis-relative mean error: {results_pelvis['summary']['mean_error']:.6f}")
        print(f"Pelvis-relative max error: {results_pelvis['summary']['max_error']:.6f}")
    else:
        print(f"Pelvis bone '{pelvis}' not found for relative validation")
    
    # Validate positions
    print("\nStep 6: Validating positions...")
    results = validator.validate_positions(derived_positions, json_positions)
    
    # Generate report
    validator.generate_report(results, pmx_path, json_path, target_frame)
    
    return results['passed']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)