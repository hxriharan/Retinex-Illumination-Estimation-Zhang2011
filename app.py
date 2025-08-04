import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
from improved_retinex import run_irr_pipeline
from ssr_msr import run_ssr, run_msr
from utils import load_image, save_image, normalize_image

def process_image(image, algorithm, **params):
    """
    Process image with selected algorithm and parameters.
    
    Args:
        image: PIL Image from Gradio
        algorithm: 'SSR', 'MSR', or 'IRR'
        **params: Algorithm-specific parameters
    
    Returns:
        Dictionary with original and processed images
    """
    if image is None:
        return None, None, None, None
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    try:
        if algorithm == "SSR":
            # Single Scale Retinex
            enhanced = run_ssr(
                img_array, 
                sigma=params.get('sigma', 80),
                normalize_method=params.get('normalize_method', 'minmax')
            )
            return image, enhanced, None, None
            
        elif algorithm == "MSR":
            # Multi Scale Retinex
            enhanced = run_msr(
                img_array,
                sigmas=params.get('sigmas', [15, 80, 250]),
                weights=params.get('weights', [1/3, 1/3, 1/3]),
                normalize_method=params.get('normalize_method', 'minmax')
            )
            return image, enhanced, None, None
            
        elif algorithm == "IRR":
            # Improved Recursive Retinex (Zhang 2011)
            results = run_irr_pipeline(
                img_array,
                illum_method=params.get('illum_method', 'gaussian'),
                illum_ksize=params.get('illum_ksize', 15),
                illum_sigma=params.get('illum_sigma', 30),
                enhance_method=params.get('enhance_method', 'sigmoid'),
                alpha=params.get('alpha', 1.0),
                beta=params.get('beta', 0.5),
                gamma=params.get('gamma', 1.0)
            )
            return results['original'], results['enhanced'], results['illumination'], results['reflectance']
        
        else:
            return image, image, None, None
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return image, image, None, None

def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Retinex Image Enhancement", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🖼️ Retinex Image Enhancement")
        gr.Markdown("Enhance images using Single Scale Retinex (SSR), Multi Scale Retinex (MSR), or Improved Recursive Retinex (IRR) algorithms.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Upload Image")
                input_image = gr.Image(label="Input Image", type="pil")
                
                # Algorithm selection
                gr.Markdown("### ⚙️ Algorithm Selection")
                algorithm = gr.Radio(
                    choices=["SSR", "MSR", "IRR"],
                    value="IRR",
                    label="Select Algorithm"
                )
                
                # Parameter controls
                gr.Markdown("### 🎛️ Parameters")
                
                with gr.Tab("SSR Parameters"):
                    ssr_sigma = gr.Slider(
                        minimum=10, maximum=200, value=80, step=5,
                        label="Sigma (Gaussian blur)"
                    )
                    ssr_normalize = gr.Radio(
                        choices=["minmax", "histogram", "adaptive"],
                        value="minmax",
                        label="Normalization Method"
                    )
                
                with gr.Tab("MSR Parameters"):
                    msr_sigma1 = gr.Slider(
                        minimum=10, maximum=100, value=15, step=5,
                        label="Sigma 1 (Small scale)"
                    )
                    msr_sigma2 = gr.Slider(
                        minimum=50, maximum=150, value=80, step=10,
                        label="Sigma 2 (Medium scale)"
                    )
                    msr_sigma3 = gr.Slider(
                        minimum=150, maximum=400, value=250, step=25,
                        label="Sigma 3 (Large scale)"
                    )
                    msr_normalize = gr.Radio(
                        choices=["minmax", "histogram", "adaptive"],
                        value="minmax",
                        label="Normalization Method"
                    )
                
                with gr.Tab("IRR Parameters"):
                    illum_method = gr.Radio(
                        choices=["gaussian", "bilateral"],
                        value="gaussian",
                        label="Illumination Method"
                    )
                    illum_ksize = gr.Slider(
                        minimum=5, maximum=51, value=15, step=2,
                        label="Kernel Size"
                    )
                    illum_sigma = gr.Slider(
                        minimum=5, maximum=100, value=30, step=5,
                        label="Sigma"
                    )
                    enhance_method = gr.Radio(
                        choices=["sigmoid", "normalize", "both"],
                        value="sigmoid",
                        label="Enhancement Method"
                    )
                    alpha = gr.Slider(
                        minimum=0.1, maximum=3.0, value=1.0, step=0.1,
                        label="Alpha (Sigmoid steepness)"
                    )
                    beta = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.1,
                        label="Beta (Sigmoid midpoint)"
                    )
                    gamma = gr.Slider(
                        minimum=0.1, maximum=3.0, value=1.0, step=0.1,
                        label="Gamma"
                    )
                
                # Process button
                process_btn = gr.Button("🚀 Process Image", variant="primary")
            
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### 📊 Results")
                
                with gr.Tab("Enhanced Image"):
                    output_enhanced = gr.Image(label="Enhanced Image", type="pil")
                
                with gr.Tab("IRR Components"):
                    with gr.Row():
                        output_illum = gr.Image(label="Illumination", type="pil")
                        output_reflect = gr.Image(label="Reflectance", type="pil")
                
                with gr.Tab("Comparison"):
                    comparison_output = gr.Image(label="Side-by-Side Comparison", type="pil")
        
        # Event handlers
        def process_with_params(image, algo, **kwargs):
            # Extract parameters based on algorithm
            params = {}
            
            if algo == "SSR":
                params.update({
                    'sigma': kwargs.get('ssr_sigma', 80),
                    'normalize_method': kwargs.get('ssr_normalize', 'minmax')
                })
            elif algo == "MSR":
                params.update({
                    'sigmas': [kwargs.get('msr_sigma1', 15), 
                              kwargs.get('msr_sigma2', 80), 
                              kwargs.get('msr_sigma3', 250)],
                    'weights': [1/3, 1/3, 1/3],
                    'normalize_method': kwargs.get('msr_normalize', 'minmax')
                })
            elif algo == "IRR":
                params.update({
                    'illum_method': kwargs.get('illum_method', 'gaussian'),
                    'illum_ksize': kwargs.get('illum_ksize', 15),
                    'illum_sigma': kwargs.get('illum_sigma', 30),
                    'enhance_method': kwargs.get('enhance_method', 'sigmoid'),
                    'alpha': kwargs.get('alpha', 1.0),
                    'beta': kwargs.get('beta', 0.5),
                    'gamma': kwargs.get('gamma', 1.0)
                })
            
            return process_image(image, algo, **params)
        
        # Connect all inputs to the process function
        inputs = [
            input_image, algorithm,
            ssr_sigma, ssr_normalize,
            msr_sigma1, msr_sigma2, msr_sigma3, msr_normalize,
            illum_method, illum_ksize, illum_sigma, enhance_method, alpha, beta, gamma
        ]
        
        outputs = [output_enhanced, output_illum, output_reflect, comparison_output]
        
        process_btn.click(
            fn=process_with_params,
            inputs=inputs,
            outputs=outputs
        )
        
        # Auto-process when image is uploaded
        input_image.change(
            fn=lambda img: process_with_params(img, "IRR"),
            inputs=[input_image],
            outputs=outputs
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, debug=True) 