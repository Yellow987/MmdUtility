# Bone Position Calculation Debug Plan

## Problem Summary

Both left and right foot bones are returning identical world space positions at frame 3000:

- LEFT_FOOT: (0.689, 34.577, 0.000)
- RIGHT_FOOT: (-0.689, 34.577, 0.000)

**Key Issues:**

1. Y and Z coordinates are identical (should be different for dancing motion)
2. Only X coordinates are mirrored (suggests some calculation is working)
3. Y=34.577 seems unusually high for foot positions
4. All foot-related bones show same Y coordinate

## Root Cause Analysis

### Hypothesis 1: VMD Animation Data Not Applied

**Symptoms:** If VMD animation data isn't being found/applied, bones would only show rest positions
**Test:** Check if `get_bone_animation_data()` returns non-zero animation values
**Evidence:** X coordinates are different, suggesting some animation data is being applied

### Hypothesis 2: Bone Hierarchy Traversal Issue

**Symptoms:** If parent bone transformations aren't properly accumulated, child bones show incorrect positions
**Test:** Check bone parent indices and transformation chain
**Evidence:** High Y values suggest possible cumulative transformation errors

### Hypothesis 3: Transformation Matrix Calculation Error

**Symptoms:** Incorrect matrix multiplication or coordinate system issues
**Test:** Verify matrix math and coordinate transformations
**Evidence:** Identical Y/Z but different X suggests partial calculation success

### Hypothesis 4: Bone Name Matching Issues

**Symptoms:** Wrong bones being calculated or fallback to default positions
**Test:** Verify Japanese/English bone name matching
**Evidence:** Both feet affected suggests systematic issue, not name-specific

## Diagnostic Steps

### Step 1: VMD Data Inspection

```python
# Check if animation data exists for foot bones
def debug_vmd_bone_data(vmd_motion, bone_names, frame=3000):
    for bone_name in bone_names:
        bone_frames = [f for f in vmd_motion.motions if f.name == bone_name]
        print(f"Bone '{bone_name}': {len(bone_frames)} keyframes")
        if bone_frames:
            # Find frames around target frame
            nearby = [f for f in bone_frames if abs(f.frame - frame) < 50]
            print(f"  Near frame {frame}: {len(nearby)} keyframes")
```

### Step 2: Bone Hierarchy Validation

```python
# Check bone parent relationships
def debug_bone_hierarchy(pmx_model, bone_name):
    bone_result = BoneHierarchyWalker.find_bone_by_name(pmx_model.bones, bone_name)
    if bone_result:
        idx, bone = bone_result
        chain = BoneHierarchyWalker.get_bone_chain_to_root(pmx_model.bones, idx)
        print(f"Bone '{bone_name}' hierarchy chain: {chain}")
        for i in chain:
            parent_bone = pmx_model.bones[i]
            print(f"  {i}: {parent_bone.name} -> parent: {parent_bone.parent_index}")
```

### Step 3: Step-by-Step Transformation Debug

```python
# Debug each transformation in the chain
def debug_transformations(pmx_model, vmd_motion, frame, bone_name):
    # Get bone chain
    bone_result = BoneHierarchyWalker.find_bone_by_name(pmx_model.bones, bone_name)
    bone_index, target_bone = bone_result
    chain = BoneHierarchyWalker.get_bone_chain_to_root(pmx_model.bones, bone_index)

    world_matrix = identity_matrix()

    for i, chain_bone_index in enumerate(chain):
        bone = pmx_model.bones[chain_bone_index]
        rest_pos = bone.position
        anim_pos, anim_quat = get_bone_animation_data(vmd_motion, bone.name, frame)

        print(f"Step {i}: Bone '{bone.name}'")
        print(f"  Rest pos: {rest_pos.x:.3f}, {rest_pos.y:.3f}, {rest_pos.z:.3f}")
        print(f"  Anim pos: {anim_pos.x:.3f}, {anim_pos.y:.3f}, {anim_pos.z:.3f}")
        print(f"  Final pos: {rest_pos.x + anim_pos.x:.3f}, {rest_pos.y + anim_pos.y:.3f}, {rest_pos.z + anim_pos.z:.3f}")

        # Apply transformation
        final_pos = Vector3(rest_pos.x + anim_pos.x, rest_pos.y + anim_pos.y, rest_pos.z + anim_pos.z)
        bone_matrix = create_transformation_matrix(final_pos, anim_quat)
        world_matrix = multiply_matrices(world_matrix, bone_matrix)

        # Show cumulative world position
        world_pos = Vector3(world_matrix[0][3], world_matrix[1][3], world_matrix[2][3])
        print(f"  World pos so far: {world_pos.x:.3f}, {world_pos.y:.3f}, {world_pos.z:.3f}")
```

### Step 4: Compare Left vs Right Foot Data

```python
# Direct comparison of left vs right foot calculations
def compare_feet_debug(pmx_model, vmd_motion, frame=3000):
    left_bones = ['左足', '左足首', '左つま先']
    right_bones = ['右足', '右足首', '右つま先']

    for left, right in zip(left_bones, right_bones):
        print(f"\nComparing {left} vs {right}:")

        # Check rest positions
        left_result = find_bone_by_name(pmx_model.bones, left)
        right_result = find_bone_by_name(pmx_model.bones, right)

        if left_result and right_result:
            _, left_bone = left_result
            _, right_bone = right_result
            print(f"  Rest positions:")
            print(f"    {left}: {left_bone.position.x:.3f}, {left_bone.position.y:.3f}, {left_bone.position.z:.3f}")
            print(f"    {right}: {right_bone.position.x:.3f}, {right_bone.position.y:.3f}, {right_bone.position.z:.3f}")

        # Check animation data
        left_anim = get_bone_animation_data(vmd_motion, left, frame)
        right_anim = get_bone_animation_data(vmd_motion, right, frame)
        print(f"  Animation data:")
        print(f"    {left}: pos={left_anim[0].x:.3f}, {left_anim[0].y:.3f}, {left_anim[0].z:.3f}")
        print(f"    {right}: pos={right_anim[0].x:.3f}, {right_anim[0].y:.3f}, {right_anim[0].z:.3f}")
```

## Suspected Issues and Fixes

### Issue 1: Matrix Multiplication Order

The current code may be multiplying matrices in wrong order:

```python
# Current (possibly wrong):
world_matrix = BoneHierarchyWalker.multiply_matrices(world_matrix, bone_matrix)

# Should it be?:
world_matrix = BoneHierarchyWalker.multiply_matrices(bone_matrix, world_matrix)
```

### Issue 2: Position Addition vs Matrix Transformation

Current code adds rest position + animation position before creating matrix:

```python
final_pos = common.Vector3(rest_pos.x + anim_pos.x, rest_pos.y + anim_pos.y, rest_pos.z + anim_pos.z)
```

This might be incorrect - animation position might need to be applied in local space first.

### Issue 3: Missing Quaternion Dot Product Method

The code calls `q1.dot(q2)` but common.Quaternion might not have this method.

### Issue 4: Bone Chain Calculation

The bone chain might not be building correctly if parent_index references are wrong.

## Test Cases to Create

1. **Simple rest position test**: Calculate positions with no animation data
2. **Single bone transformation**: Test one bone in isolation
3. **Animation data extraction**: Verify VMD keyframe matching
4. **Matrix math verification**: Test matrix multiplication with known values
5. **Coordinate system check**: Ensure right-handed/left-handed consistency

## Expected Outcomes After Fix

- Left and right foot should show clearly different Y coordinates
- Foot positions should be reasonable (likely Y < 10 for ground contact)
- Bone hierarchy should show proper parent-child relationships
- Animation data should show non-zero values for dancing motion
