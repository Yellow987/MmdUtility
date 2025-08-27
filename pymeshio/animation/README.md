# PyMeshIO Animation Module

The animation module provides functionality to combine PMX models with VMD motion data and query bone world positions at specific frames using PyTorch-based forward kinematics.

## Features

- **AnimatedModel**: High-level class that combines PMX and VMD data
- **ForwardKinematics**: Efficient PyTorch-based FK calculations
- **Frame-based Queries**: Get world positions of all bones at any frame
- **Caching**: Performance optimization for repeated queries
- **MMD Coordinate System**: All calculations performed in native MMD coordinates

## Quick Start

```python
from pymeshio.animation import create_animated_model

# Create animated model from files
animated_model = create_animated_model("model.pmx", "motion.vmd")

# Get world positions of all bones at frame 30
world_positions = animated_model.get_world_positions(30)

# Get position of a specific bone
center_pos = animated_model.get_bone_world_position("センター", 30)

# Get available frames and bone names
frames = animated_model.get_available_frames()
bones = animated_model.get_bone_names()
```

## Detailed Usage

### Creating an AnimatedModel

#### Method 1: Using convenience function

```python
from pymeshio.animation import create_animated_model

animated_model = create_animated_model(
    pmx_path="path/to/model.pmx",
    vmd_path="path/to/motion.vmd",
    device='cpu'  # or 'cuda' for GPU acceleration
)
```

#### Method 2: Using existing PMX/VMD objects

```python
from pymeshio.animation import AnimatedModel
import pymeshio.pmx.reader as pmx_reader
import pymeshio.vmd.reader as vmd_reader

# Load PMX and VMD separately
pmx_model = pmx_reader.read_from_file("model.pmx")
vmd_motion = vmd_reader.read_from_file("motion.vmd")

# Create animated model
animated_model = AnimatedModel(pmx_model, vmd_motion, device='cpu')
```

### Querying World Positions

#### Get all bone positions at a frame

```python
# Get world positions for all bones at frame 60
world_positions = animated_model.get_world_positions(60)

# Result is a dictionary: bone_name -> numpy array [x, y, z]
for bone_name, position in world_positions.items():
    print(f"{bone_name}: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
```

#### Get specific bone position

```python
# Get position of center bone at frame 60
center_pos = animated_model.get_bone_world_position("センター", 60)

if center_pos is not None:
    print(f"Center: [{center_pos[0]:.3f}, {center_pos[1]:.3f}, {center_pos[2]:.3f}]")
```

#### Performance with caching

```python
# First call calculates and caches the result
positions_1 = animated_model.get_world_positions(30, use_cache=True)

# Second call uses cached result (much faster)
positions_2 = animated_model.get_world_positions(30, use_cache=True)

# Disable caching for memory-constrained environments
positions_3 = animated_model.get_world_positions(30, use_cache=False)

# Clear cache to free memory
animated_model.clear_cache()
```

### Getting Model Information

```python
# Get all bone names in the model
bone_names = animated_model.get_bone_names()
print(f"Model has {len(bone_names)} bones")

# Get all available animation frames
frames = animated_model.get_available_frames()
print(f"Animation has {len(frames)} frames: {frames[:10]}...")

# Get rest position of a bone (in MMD coordinates)
rest_pos = animated_model.get_rest_position("センター")
print(f"Center rest position: {rest_pos}")
```

## Coordinate System

The animation module uses **MMD coordinates** throughout:

- **X**: Right (positive) / Left (negative)
- **Y**: Up (positive) / Down (negative)
- **Z**: Forward (positive) / Backward (negative)
- **Units**: Native MMD units (typically ~0.08m scale)

### Coordinate System Notes

1. **PMX rest positions** are stored in MMD coordinates as absolute world positions
2. **VMD motion data** is kept in native MMD coordinates
3. **Forward kinematics calculations** use relative bone offsets computed from PMX data
4. **Output world positions** are in MMD coordinates

This ensures consistency and avoids coordinate transformation errors that can occur when mixing coordinate systems.

## Architecture

### Class Hierarchy

```
AnimatedModel
├── ForwardKinematics (PyTorch-based FK calculations)
├── PMX Model (bone hierarchy, rest positions)
├── VMD Motion (per-frame bone rotations/translations)
└── Cache (performance optimization)
```

### Data Flow

1. **Initialization**: Process PMX bone hierarchy and VMD motion data
2. **Query**: Convert frame data to PyTorch tensors
3. **FK Calculation**: Compute world positions using forward kinematics
4. **Caching**: Store results for repeated queries
5. **Output**: Return positions as numpy arrays in dictionary

## Performance Considerations

### GPU Acceleration

```python
# Use GPU for faster calculations on large models
animated_model = create_animated_model("model.pmx", "motion.vmd", device='cuda')
```

### Memory Management

```python
# Clear cache periodically in long-running applications
animated_model.clear_cache()

# Disable caching for memory-constrained environments
world_positions = animated_model.get_world_positions(frame, use_cache=False)
```

### Batch Processing

```python
# Process multiple frames efficiently
frames_to_process = [30, 60, 90, 120]
all_results = {}

for frame in frames_to_process:
    all_results[frame] = animated_model.get_world_positions(frame)
```

## Error Handling

The module includes comprehensive error handling:

```python
try:
    animated_model = create_animated_model("model.pmx", "motion.vmd")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid model data: {e}")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

## Requirements

- **PyTorch**: Required for forward kinematics calculations
- **NumPy**: For numerical operations
- **Python 3.6+**: Modern Python version

## Examples

See `test_animation.py` for complete working examples of:

- Basic animation functionality
- Performance testing with caching
- Error handling
- GPU acceleration (if available)

## Integration

The animation module integrates seamlessly with existing pymeshio code:

```python
# Use with existing PMX/VMD loading code
import pymeshio
import pymeshio.animation

# Load and process models
pmx_model = pymeshio.pmx.reader.read_from_file("model.pmx")
vmd_motion = pymeshio.vmd.reader.read_from_file("motion.vmd")

# Add animation capabilities
animated_model = pymeshio.animation.AnimatedModel(pmx_model, vmd_motion)
world_positions = animated_model.get_world_positions(60)
```
