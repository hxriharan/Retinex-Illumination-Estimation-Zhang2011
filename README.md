# Recursive Retinex Implementation (Zhang 2011)

This repository implements the Retinex-based image enhancement algorithm from the paper **"Retinex-based Image Enhancement for Illumination and Reflectance Estimation"** (Zhang et al., 2011).

## Project Structure

```
recursive_retinex/
├── core.py                 # Core algorithms (recursive filtering, 8-dir processing)
├── tone_mapping.py         # Sigmoid tone compression and normalization
├── run_pipeline.py         # Main script to load image, run IRR, and save/display output
└── __init__.py            # Package initialization and exports
```

## Features

### Core Algorithm
- **Illumination Estimation**: Gaussian and bilateral filtering methods
- **Reflectance Recovery**: Log-domain subtraction
- **Recursive Filtering**: 8-directional processing (placeholder for Zhang 2011 implementation)
- **Gradient Computation**: Finite difference methods for edge detection

### Tone Mapping
- **Sigmoid Compression**: Non-linear tone mapping for contrast enhancement
- **Histogram Equalization**: Multiple normalization methods
- **Gamma Correction**: Adjustable brightness/contrast

### Pipeline
- **Complete Workflow**: From image loading to enhanced output
- **Parameter Tuning**: Adjustable filtering and enhancement parameters
- **Visualization**: Side-by-side comparison of original, illumination, reflectance, and enhanced images

## Usage

### Basic Usage

```python
from recursive_retinex import run_retinex_pipeline, display_results

# Run the complete pipeline
results = run_retinex_pipeline(
    image_path='path/to/image.jpg',
    illum_method='gaussian',  # or 'bilateral'
    illum_ksize=15,
    illum_sigma=30,
    enhance_method='sigmoid',
    alpha=1.0,
    beta=0.5
)

# Display results
display_results(results)
```

### Advanced Usage

```python
from recursive_retinex import (
    rgb_to_log, estimate_illumination, recover_reflectance,
    enhance_contrast, log_to_rgb
)

# Manual step-by-step processing
image = load_image('path/to/image.jpg')
log_image = rgb_to_log(image)
log_illum = estimate_illumination(log_image, method='bilateral')
log_reflect = recover_reflectance(log_image, log_illum)
reflectance = log_to_rgb(log_reflect)
enhanced = enhance_contrast(reflectance, method='sigmoid', alpha=1.5, beta=0.3)
```

## Parameters

### Illumination Estimation
- `method`: 'gaussian' or 'bilateral'
- `ksize`: Kernel size for Gaussian blur (default: 15)
- `sigma`: Sigma for Gaussian blur (default: 30)

### Tone Mapping
- `alpha`: Sigmoid steepness (default: 1.0)
- `beta`: Sigmoid midpoint (default: 0.5)
- `gamma`: Gamma correction (default: 1.0)

### Recursive Filtering (Future)
- `num_iterations`: Number of recursive iterations
- `lambda_param`: Smoothing parameter

## Future Enhancements

### Planned UI (Gradio)
- **Upload Interface**: Drag-and-drop image upload
- **Real-time Preview**: Show original, illumination, reflectance, and enhanced images
- **Parameter Sliders**: Interactive tuning of all parameters
- **Batch Processing**: Process multiple images at once

### Algorithm Improvements
- **True 8-directional Recursive Filtering**: Implement the specific algorithm from Zhang 2011
- **Poisson Solver**: For more accurate illumination estimation
- **Multi-scale Processing**: Handle different image scales
- **Color Space Optimization**: Better handling of different color spaces

## Dependencies

- OpenCV (cv2)
- NumPy
- Matplotlib
- (Future: Gradio for UI)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Retinex_Project

# Install dependencies
pip install opencv-python numpy matplotlib

# Run the pipeline
python recursive_retinex/run_pipeline.py
```

## References

- Zhang, M., et al. "Retinex-based Image Enhancement for Illumination and Reflectance Estimation." ICWAPR 2011.
- Land, E.H. "The Retinex Theory of Color Vision." Scientific American, 1977.

