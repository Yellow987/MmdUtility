# Pytransform3d Skeleton Loading Implementation Plan

## Question: Will TransformManager work with MMD bones?

**✅ YES! TransformManager is perfect for MMD bones because:**

### Non-Connected Bones Support

- MMD bones don't need to connect tail-to-head (head can be offset from neck origin)
- TransformManager creates coordinate frames with arbitrary position offsets
- Each bone gets its own coordinate frame positioned relative to its parent
- Example: Head bone offset (0, 2.5, 0.8) from neck → TransformManager handles this perfectly

### Position + Rotation Animation Support

- VMD animations contain both translation AND rotation per bone per frame
- TransformManager applies both as transformation matrices automatically
- Final transform: `parent_world_transform × local_rest_transform × animation_transform`
- Handles hierarchical propagation: parent moves → all children follow correctly

## Implementation Plan: Load PMX Skeleton into TransformManager

### Step 1: Add TransformManager to BonePositionExtractor

**Add to [`bone_position_extractor.py`](bone_position_extractor.py):**

```python
try:
    from pytransform3d.transformations import TransformManager
    HAS_PYTRANSFORM3D = True
except ImportError:
    HAS_PYTRANSFORM3D = False

class BonePositionExtractor:
    def __init__(self):
        # existing code...
        self.transform_manager = None

    def _initialize_transform_manager(self):
        """Initialize TransformManager for skeleton."""
        if not HAS_PYTRANSFORM3D:
            raise ImportError("pytransform3d required for skeleton transforms")
        self.transform_manager = TransformManager()
```

### Step 2: Load Skeleton Method

**`_load_skeleton_to_transform_manager()`:**

```python
def _load_skeleton_to_transform_manager(self):
    """Load PMX skeleton into TransformManager with rest pose transforms."""
    if not self.pmx_model or not self.bone_hierarchy:
        raise ValueError("PMX model must be loaded first")

    self._initialize_transform_manager()

    # Add world coordinate frame
    self.transform_manager.add_transform("world", "world", np.eye(4))

    # Process bones in dependency order (parents before children)
    processed_bones = set()

    def process_bone(bone_index):
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
        transform_matrix[:3, 3] = local_pos  # Set translation

        # Add transform to manager
        parent_frame = "world" if parent_idx == -1 else f"bone_{parent_idx}"
        self.transform_manager.add_transform(
            parent_frame, bone_name, transform_matrix
        )

        processed_bones.add(bone_index)

    # Process all bones
    for bone_index in self.bone_hierarchy:
        process_bone(bone_index)
```

### Step 3: Get Rest Pose via TransformManager

**`get_rest_pose_positions_pytransform3d()`:**

```python
def get_rest_pose_positions_pytransform3d(self):
    """Get rest pose positions using TransformManager."""
    if not self.transform_manager:
        self._load_skeleton_to_transform_manager()

    positions = {}

    for bone_index, bone_info in self.bone_hierarchy.items():
        bone_name = bone_info['name']
        frame_name = f"bone_{bone_index}"

        # Get world transform for this bone
        world_transform = self.transform_manager.get_transform(
            "world", frame_name
        )

        # Extract position (translation part)
        world_pos = world_transform[:3, 3]
        positions[bone_name] = tuple(world_pos)

    return positions
```

### Step 4: Test Implementation

**Create `test_pytransform3d_integration.py`:**

```python
#!/usr/bin/env python3
"""Test pytransform3d skeleton loading."""

def test_skeleton_loading():
    """Test loading PMX skeleton into TransformManager."""
    print("Testing pytransform3d skeleton loading...")

    pmx_path = "test/pdtt.pmx"
    if not os.path.exists(pmx_path):
        print(f"Test file not found: {pmx_path}")
        return False

    extractor = BonePositionExtractor()
    extractor.load_pmx(pmx_path)

    # Test TransformManager loading
    try:
        extractor._load_skeleton_to_transform_manager()
        print("✓ Skeleton loaded into TransformManager")
    except Exception as e:
        print(f"✗ Failed to load skeleton: {e}")
        return False

    # Compare rest poses
    print("\nComparing rest pose results...")

    # Current implementation
    current_positions = extractor.get_rest_pose_positions()

    # TransformManager implementation
    tm_positions = extractor.get_rest_pose_positions_pytransform3d()

    # Compare results
    matching_bones = 0
    total_bones = len(current_positions)

    for bone_name, current_pos in current_positions.items():
        if bone_name in tm_positions:
            tm_pos = tm_positions[bone_name]
            # Check if positions match (within tolerance)
            diff = np.linalg.norm(np.array(current_pos) - np.array(tm_pos))
            if diff < 0.001:  # 1mm tolerance
                matching_bones += 1
            else:
                print(f"  Diff {bone_name}: {current_pos} vs {tm_pos} (diff: {diff:.6f})")

    print(f"✓ {matching_bones}/{total_bones} bones match between implementations")

    return matching_bones == total_bones
```

## Implementation Summary

### What We'll Build

1. **TransformManager Integration**: Add pytransform3d support to BonePositionExtractor
2. **Skeleton Loading**: Load PMX bone hierarchy into TransformManager coordinate frames
3. **Rest Pose Validation**: Compare TransformManager results with current implementation
4. **Test Script**: Validate the integration works correctly

### Testing Strategy

1. **Load Test PMX**: Use existing `test/pdtt.pmx` file
2. **Compare Results**: TransformManager vs current `get_rest_pose_positions()`
3. **Position Accuracy**: Verify bone positions match within tolerance
4. **Hierarchy Validation**: Ensure parent-child relationships work

### Success Criteria

- ✅ TransformManager loads skeleton without errors
- ✅ Rest pose positions match current implementation (within 1mm tolerance)
- ✅ All bones are correctly positioned in hierarchy
- ✅ Ready for next phase: VMD animation application

This focused implementation gets the skeleton loading working and validates it against known-good results from the current rest pose calculation.
