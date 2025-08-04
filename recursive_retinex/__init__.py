"""
Recursive Retinex Implementation based on Zhang 2011.

This package implements the Retinex-based image enhancement algorithm
from the paper "Retinex-based Image Enhancement for Illumination and Reflectance Estimation"
by Zhang et al., 2011.
"""

from .core import (
    rgb_to_log,
    log_to_rgb,
    compute_gradients,
    estimate_illumination,
    recover_reflectance,
    recursive_filter_8dir
)

from .tone_mapping import (
    sigmoid_tone_compression,
    normalize_image,
    enhance_contrast,
    adjust_gamma
)

from .run_pipeline import (
    load_image,
    save_image,
    run_retinex_pipeline,
    display_results
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Core functions
    'rgb_to_log',
    'log_to_rgb',
    'compute_gradients',
    'estimate_illumination',
    'recover_reflectance',
    'recursive_filter_8dir',
    
    # Tone mapping functions
    'sigmoid_tone_compression',
    'normalize_image',
    'enhance_contrast',
    'adjust_gamma',
    
    # Pipeline functions
    'load_image',
    'save_image',
    'run_retinex_pipeline',
    'display_results'
] 