# coding: utf-8
"""
Animation module for pymeshio

Provides functionality to combine PMX models with VMD motion data
and query bone world positions at specific frames using PyTorch-based
forward kinematics.

Includes both single-file and batch processing capabilities:
- AnimatedModel: Single VMD file processing
- BatchAnimatedModel: Efficient batch processing of multiple VMD files
"""

from .animated_model import AnimatedModel, create_animated_model
from .forward_kinematics import ForwardKinematics
from .batch_forward_kinematics import BatchForwardKinematics, create_bone_filter_indices
from .batch_animated_model import BatchAnimatedModel, create_batch_animated_model

__all__ = [
    'AnimatedModel', 'create_animated_model',
    'ForwardKinematics',
    'BatchForwardKinematics', 'create_bone_filter_indices',
    'BatchAnimatedModel', 'create_batch_animated_model'
]