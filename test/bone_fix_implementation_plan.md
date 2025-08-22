# Bone Position Calculation Fix Implementation Plan

## Summary

The bone position calculation is failing because [`bone_animation.py`](MmdUtility/bone_animation.py) reimplements math functionality that already exists in [`pymeshio.common`](MmdUtility/pymeshio/common.py), causing incompatibilities that result in identical Y coordinates for left and right feet.

## Implementation Strategy: Use Existing Infrastructure

**Approach**: Refactor [`bone_animation.py`](MmdUtility/bone_animation.py) to use the existing [`common.Quaternion`](MmdUtility/pymeshio/common.py:207-309) and [`common.Vector3`](MmdUtility/pymeshio/common.py:60-133) classes instead of custom implementations.

## Specific Changes Required

### 1. Import and Dependency Updates

**File**: [`bone_animation.py`](MmdUtility/bone_animation.py:20-21)

```python
# ADD:
import numpy as np

# MODIFY existing imports to ensure common classes are available
from pymeshio import common, pmx, vmd
```

### 2. Remove Custom Quaternion Math Methods

**File**: [`bone_animation.py`](MmdUtility/bone_animation.py:162-206)

- **DELETE**: `slerp_quaternion()` method (lines 162-206)
- **REPLACE**: Use existing quaternion operations or simple linear interpolation

### 3. Replace Custom Matrix Operations

**File**: [`bone_animation.py`](MmdUtility/bone_animation.py:264-332)

- **DELETE**: `create_transformation_matrix()` (lines 264-293)
- **DELETE**: `multiply_matrices()` (lines 295-314)
- **DELETE**: `apply_matrix_to_point()` (lines 316-332)
- **REPLACE**: Use [`common.Quaternion.getMatrix()`](MmdUtility/pymeshio/common.py:231-250) and numpy operations

### 4. Fix Core Position Calculation Logic

**File**: [`bone_animation.py`](MmdUtility/bone_animation.py:394-476)

**Current problematic code** (lines 450-475):

```python
# Apply transformations from root to target
for chain_bone_index in bone_chain:
    bone = pmx_model.bones[chain_bone_index]

    # Get bone's rest position
    rest_pos = bone.position

    # Get animation data for this bone at the target frame
    anim_pos, anim_quat = get_bone_animation_data(vmd_motion, bone.name, frame_number)

    # Combine rest pose and animation
    final_pos = common.Vector3(
        rest_pos.x + anim_pos.x,
        rest_pos.y + anim_pos.y,
        rest_pos.z + anim_pos.z
    )

    # Create transformation matrix for this bone
    bone_matrix = BoneHierarchyWalker.create_transformation_matrix(final_pos, anim_quat)

    # Accumulate transformation
    world_matrix = BoneHierarchyWalker.multiply_matrices(world_matrix, bone_matrix)

# Extract world position from final transformation matrix
world_pos = common.Vector3(world_matrix[0][3], world_matrix[1][3], world_matrix[2][3])
```

**New fixed code**:

```python
# Start with identity transformation
world_transform = np.eye(4, dtype=float)

# Apply transformations from root to target
for chain_bone_index in bone_chain:
    bone = pmx_model.bones[chain_bone_index]

    # Get bone's rest position
    rest_pos = bone.position

    # Get animation data for this bone at the target frame
    anim_pos, anim_quat = get_bone_animation_data(vmd_motion, bone.name, frame_number)

    # Create translation matrix for rest position
    rest_translation = np.eye(4, dtype=float)
    rest_translation[0:3, 3] = [rest_pos.x, rest_pos.y, rest_pos.z]

    # Create translation matrix for animation position
    anim_translation = np.eye(4, dtype=float)
    anim_translation[0:3, 3] = [anim_pos.x, anim_pos.y, anim_pos.z]

    # Get rotation matrix from quaternion (using existing common.Quaternion)
    rotation_matrix = anim_quat.getMatrix()  # This returns numpy array

    # Combine: Translation * Rotation * Rest_Translation
    bone_transform = np.dot(np.dot(anim_translation, rotation_matrix), rest_translation)

    # Accumulate transformation (matrix multiplication order is important!)
    world_transform = np.dot(world_transform, bone_transform)

# Extract world position from final transformation matrix
world_pos = world_transform[0:3, 3]  # Get translation component
```

### 5. Fix VMD Interpolation Method

**File**: [`bone_animation.py`](MmdUtility/bone_animation.py:121-160)

**Current code** (lines 157-158):

```python
# Interpolate rotation using spherical linear interpolation (slerp)
t_rot = VMDInterpolator.bezier_interpolate(t, bezier_data['rot'])
quat = VMDInterpolator.slerp_quaternion(frame1.q, frame2.q, t_rot)
```

**New code**:

```python
# Interpolate rotation - use simpler linear interpolation for now
t_rot = VMDInterpolator.bezier_interpolate(t, bezier_data['rot'])

# Simple linear quaternion interpolation (can be improved later)
quat = common.Quaternion(
    frame1.q.x + (frame2.q.x - frame1.q.x) * t_rot,
    frame1.q.y + (frame2.q.y - frame1.q.y) * t_rot,
    frame1.q.z + (frame2.q.z - frame1.q.z) * t_rot,
    frame1.q.w + (frame2.q.w - frame1.q.w) * t_rot
).getNormalized()  # Use existing normalization method
```

### 6. Add Coordinate System Handling

**Consider using**: [`common.Quaternion.getRHMatrix()`](MmdUtility/pymeshio/common.py:252-275) if coordinate system issues persist.

## Implementation Steps

1. **Phase 1: Minimal Fix**

   - Replace matrix operations with numpy-based implementation
   - Use [`common.Quaternion.getMatrix()`](MmdUtility/pymeshio/common.py:231-250) for rotation matrices
   - Fix transformation chain accumulation

2. **Phase 2: Test and Verify**

   - Run test with fixed implementation
   - Verify left/right feet show different Y coordinates
   - Check if positions look reasonable (Y < 10 for ground contact)

3. **Phase 3: Refinement**
   - Add proper quaternion interpolation if needed
   - Handle coordinate system conversion if required
   - Optimize performance

## Expected Results After Fix

- **Left and right feet**: Should show clearly different Y coordinates
- **Position values**: Should be reasonable (likely Y < 10 for foot contact)
- **Animation consistency**: Bone positions should change smoothly between frames
- **Hierarchy correctness**: Child bones should move with their parents

## Validation Tests

1. **Basic test**: Single frame foot positions should differ
2. **Animation test**: Positions should change between frames 2990-3010
3. **Hierarchy test**: Parent bone changes should affect children
4. **Symmetry test**: Left/right should be mirrored in X but different in Y/Z

This fix addresses the fundamental math compatibility issues that are causing identical bone positions.
