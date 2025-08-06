import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
from improved_retinex import run_irr_pipeline
from ssr_msr import single_scale_retinex, multi_scale_retinex
from utils import load_image, save_image, normalize_image

def process_image_all_algorithms(image):
    """
    Process image with all three algorithms (SSR, MSR, IRR) and return results.
    
    Args:
        image: PIL Image from Gradio
    
    Returns:
        Tuple of (original, ssr_result, msr_result, irr_result)
    """
    if image is None:
        return None, None, None, None
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    try:
        # Run SSR (Single Scale Retinex)
        ssr_result = single_scale_retinex(img_array, sigma=80)
        # Convert from [0,1] to [0,255] for display
        ssr_result = (ssr_result * 255).astype(np.uint8)
        
        # Run MSR (Multi Scale Retinex)
        msr_result = multi_scale_retinex(img_array, sigmas=[15, 80, 250])
        # Convert from [0,1] to [0,255] for display
        msr_result = (msr_result * 255).astype(np.uint8)
        
        # Run IRR (Improved Recursive Retinex) with fixed threshold
        irr_results = run_irr_pipeline(
            img_array,
            illum_method='gaussian',
            illum_ksize=15,
            illum_sigma=30,
            enhance_method='sigmoid',
            alpha=1.0,
            beta=0.1,  # Fixed threshold at 0.1 as requested
            gamma=1.0
        )
        irr_result = irr_results['enhanced']
        
        return image, ssr_result, msr_result, irr_result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return image, image, image, image

def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Retinex Image Enhancement", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Retinex Image Enhancement")
        gr.Markdown("Upload an image to see enhanced results using Single Scale Retinex (SSR), Multi Scale Retinex (MSR), and Improved Recursive Retinex (IRR) algorithms.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Upload Image")
                input_image = gr.Image(label="Input Image", type="pil")
                
                # Process button
                process_btn = gr.Button("Process Image", variant="primary")
                
                # Information about algorithms
                gr.Markdown("### Algorithm Information")
                gr.Markdown("""
                **SSR (Single Scale Retinex)**: Uses Gaussian blur with sigma=80
                
                **MSR (Multi Scale Retinex)**: Uses three scales [15, 80, 250] with equal weights
                
                **IRR (Improved Recursive Retinex)**: Based on Zhang 2011 with fixed threshold β=0.1
                """)
            
            with gr.Column(scale=4):
                # Output section
                gr.Markdown("### Results")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Original")
                        output_original = gr.Image(label="Original Image", type="pil")
                    
                    with gr.Column():
                        gr.Markdown("#### SSR Result")
                        output_ssr = gr.Image(label="SSR Enhanced", type="pil")
                    
                    with gr.Column():
                        gr.Markdown("#### MSR Result")
                        output_msr = gr.Image(label="MSR Enhanced", type="pil")
                    
                    with gr.Column():
                        gr.Markdown("#### IRR Result")
                        output_irr = gr.Image(label="IRR Enhanced", type="pil")
        
        # Event handlers
        def process_image(image):
            return process_image_all_algorithms(image)
        
        # Connect inputs to outputs
        inputs = [input_image]
        outputs = [output_original, output_ssr, output_msr, output_irr]
        
        process_btn.click(
            fn=process_image,
            inputs=inputs,
            outputs=outputs
        )
        
        # Auto-process when image is uploaded
        input_image.change(
            fn=process_image,
            inputs=inputs,
            outputs=outputs
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, debug=True) 