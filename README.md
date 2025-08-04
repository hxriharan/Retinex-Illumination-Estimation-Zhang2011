# Recursive Retinex Implementation (Zhang 2011)

This repository implements various Retinex-based image enhancement algorithms, including Single Scale Retinex (SSR), Multi Scale Retinex (MSR), and Improved Recursive Retinex (IRR) based on Zhang 2011.

## Project Structure

```
Retinex_Project/
├── app.py                  ← Gradio UI with interactive parameter tuning
├── ssr_msr.py             ← Single Scale Retinex (SSR) and Multi Scale Retinex (MSR) implementations
├── improved_retinex.py    ← Improved Recursive Retinex (IRR) implementation (Zhang 2011)
├── utils.py               ← Image loading, normalization, and color correction utilities
├── Files/                 ← Test images and output directory
└── README.md              ← This documentation
```

## Features

### Interactive UI (Gradio)
- **Drag-and-drop image upload**
- **Real-time parameter tuning** with sliders
- **Side-by-side comparison** of original, enhanced, illumination, and reflectance
- **Multiple algorithm support**: SSR, MSR, and IRR
- **Export capabilities** for processed images

### Algorithm Implementations

#### Single Scale Retinex (SSR)
- Gaussian blur-based illumination estimation
- Adjustable sigma parameter
- Multiple normalization methods

#### Multi Scale Retinex (MSR)
- Multi-scale processing with configurable sigmas
- Weighted combination of different scales
- Enhanced detail preservation

#### Improved Recursive Retinex (IRR)
- Based on Zhang et al., 2011
- Gaussian and bilateral filtering options
- Sigmoid tone compression
- Gamma correction
- 8-directional recursive filtering (placeholder for full implementation)

### Utility Functions
- **Image loading and saving**
- **Multiple normalization methods** (minmax, histogram, adaptive)
- **Color correction** (white balance, gamma)
- **Image resizing and comparison**
- **Quality metrics** (MSE, PSNR, SSIM)

This will launch a Gradio interface where you can:
1. Upload an image
2. Select an algorithm (SSR, MSR, or IRR)
3. Adjust parameters in real-time
4. View results and download enhanced images

This project is open source. Please check the license file for details.

