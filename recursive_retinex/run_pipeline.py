import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from core import (
    rgb_to_log, log_to_rgb, estimate_illumination, 
    recover_reflectance, recursive_filter_8dir
)
from tone_mapping import enhance_contrast, adjust_gamma

def load_image(image_path):
    """Load and preprocess image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_image(img, path):
    """Save image to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def run_retinex_pipeline(image_path, output_dir='Files/outputs', 
                        illum_method='gaussian', illum_ksize=15, illum_sigma=30,
                        use_recursive=False, num_iterations=3, lambda_param=0.1,
                        enhance_method='sigmoid', alpha=1.0, beta=0.5, gamma=1.0):
    """
    Run the complete Retinex pipeline.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        illum_method: 'gaussian' or 'bilateral' for illumination estimation
        illum_ksize: Kernel size for Gaussian blur
        illum_sigma: Sigma for Gaussian blur
        use_recursive: Whether to use recursive filtering
        num_iterations: Number of recursive iterations
        lambda_param: Smoothing parameter for recursive filtering
        enhance_method: Method for contrast enhancement
        alpha: Sigmoid alpha parameter
        beta: Sigmoid beta parameter
        gamma: Gamma correction value
    
    Returns:
        Dictionary containing all processed images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    log_image = rgb_to_log(image)
    
    # Estimate illumination
    print("Estimating illumination...")
    if use_recursive:
        log_illum = recursive_filter_8dir(log_image, num_iterations, lambda_param)
    else:
        log_illum = estimate_illumination(log_image, illum_method, illum_ksize, illum_sigma)
    
    # Recover reflectance
    print("Recovering reflectance...")
    log_reflect = recover_reflectance(log_image, log_illum)
    reflectance = log_to_rgb(log_reflect)
    illum_rgb = log_to_rgb(log_illum)
    
    # Enhance contrast
    print("Enhancing contrast...")
    enhanced = enhance_contrast(reflectance, method=enhance_method, alpha=alpha, beta=beta)
    
    # Apply gamma correction if needed
    if gamma != 1.0:
        enhanced = adjust_gamma(enhanced, gamma)
    
    # Save outputs
    print("Saving outputs...")
    save_image(reflectance, os.path.join(output_dir, 'reflectance.png'))
    save_image(illum_rgb, os.path.join(output_dir, 'illumination.png'))
    save_image(enhanced, os.path.join(output_dir, 'enhanced.png'))
    
    return {
        'original': image,
        'illumination': illum_rgb,
        'reflectance': reflectance,
        'enhanced': enhanced
    }

def display_results(results, save_plot=True, output_dir='Files/outputs'):
    """Display and optionally save comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original
    axes[0, 0].imshow(results['original'])
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Illumination
    axes[0, 1].imshow(results['illumination'])
    axes[0, 1].set_title('Estimated Illumination')
    axes[0, 1].axis('off')
    
    # Reflectance
    axes[1, 0].imshow(results['reflectance'])
    axes[1, 0].set_title('Reflectance')
    axes[1, 0].axis('off')
    
    # Enhanced
    axes[1, 1].imshow(results['enhanced'])
    axes[1, 1].set_title('Enhanced')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = os.path.join(output_dir, 'comparison.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Saved comparison plot: {plot_path}")
    
    plt.show()

def main():
    """Main function to run the pipeline."""
    # Example usage
    image_path = 'Files/test.jpg'
    
    # Run with Gaussian filtering
    print("Running pipeline with Gaussian filtering...")
    results_gaussian = run_retinex_pipeline(
        image_path=image_path,
        illum_method='gaussian',
        illum_ksize=15,
        illum_sigma=30,
        enhance_method='sigmoid',
        alpha=1.0,
        beta=0.5
    )
    
    # Display results
    display_results(results_gaussian)
    
    # Run with bilateral filtering
    print("\nRunning pipeline with bilateral filtering...")
    results_bilateral = run_retinex_pipeline(
        image_path=image_path,
        illum_method='bilateral',
        enhance_method='sigmoid',
        alpha=1.0,
        beta=0.5
    )
    
    # Display results
    display_results(results_bilateral, output_dir='Files/outputs_bilateral')

if __name__ == "__main__":
    main() 