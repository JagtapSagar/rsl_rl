# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Definitions for components of modules."""

from .memory import Memory
from .mlp import MLP
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization

# Encoder-Decoder components
from .encoders import BaseEncoder, ProprioceptionEncoder, VisionEncoder, PrivilegedEncoder
from .fusion import BaseFusion, ConcatFusion, AttentionFusion, GatedFusion
from .decoders import BaseDecoder, ActionDecoder, ValueDecoder, PolicyDecoder
from .encoder_decoder import EncoderDecoderNetwork

__all__ = [
    "MLP", "Memory", "EmpiricalNormalization", "EmpiricalDiscountedVariationNormalization",
    # Encoder-Decoder exports
    "BaseEncoder", "ProprioceptionEncoder", "VisionEncoder", "PrivilegedEncoder",
    "BaseFusion", "ConcatFusion", "AttentionFusion", "GatedFusion",
    "BaseDecoder", "ActionDecoder", "ValueDecoder", "PolicyDecoder",
    "EncoderDecoderNetwork"
]
