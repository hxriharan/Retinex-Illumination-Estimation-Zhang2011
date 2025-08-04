import cv2
import numpy as np
from utils import normalize_image

def rgb_to_log(img, epsilon=1e-6):
    """Convert RGB image to log domain."""
    img = img.astype(np.float32) / 255.0
    return np.log(img + epsilon)

def log_to_rgb(log_img):
    """Convert log domain image back to RGB."""
    img = np.exp(log_img)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def single_scale_retinex(img, sigma=80):
    """
    Single Scale Retinex algorithm.
    
    Args:
        img: Input RGB image
        sigma: Gaussian blur sigma
    
    Returns:
        Enhanced image in log domain
    """
    # Convert to log domain
    log_img = rgb_to_log(img)
    
    # Apply Gaussian blur to estimate illumination
    illum = np.zeros_like(log_img)
    for c in range(3):
        illum[:, :, c] = cv2.GaussianBlur(log_img[:, :, c], (0, 0), sigma)
    
    # Recover reflectance
    reflectance = log_img - illum
    
    return reflectance

def multi_scale_retinex(img, sigmas=[15, 80, 250], weights=None):
    """
    Multi Scale Retinex algorithm.
    
    Args:
        img: Input RGB image
        sigmas: List of Gaussian blur sigmas for different scales
        weights: Weights for each scale (default: equal weights)
    
    Returns:
        Enhanced image in log domain
    """
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    
    # Convert to log domain
    log_img = rgb_to_log(img)
    
    # Apply SSR for each scale
    reflectance_sum = np.zeros_like(log_img)
    
    for sigma, weight in zip(sigmas, weights):
        # Apply Gaussian blur
        illum = np.zeros_like(log_img)
        for c in range(3):
            illum[:, :, c] = cv2.GaussianBlur(log_img[:, :, c], (0, 0), sigma)
        
        # Recover reflectance for this scale
        reflectance = log_img - illum
        
        # Weighted sum
        reflectance_sum += weight * reflectance
    
    return reflectance_sum

def run_ssr(img, sigma=80, normalize_method='minmax'):
    """
    Run Single Scale Retinex with normalization.
    
    Args:
        img: Input RGB image
        sigma: Gaussian blur sigma
        normalize_method: Normalization method
    
    Returns:
        Enhanced image
    """
    # Apply SSR
    reflectance = single_scale_retinex(img, sigma)
    
    # Convert back to RGB
    enhanced = log_to_rgb(reflectance)
    
    # Normalize
    enhanced = normalize_image(enhanced, method=normalize_method)
    
    return enhanced

def run_msr(img, sigmas=[15, 80, 250], weights=None, normalize_method='minmax'):
    """
    Run Multi Scale Retinex with normalization.
    
    Args:
        img: Input RGB image
        sigmas: List of Gaussian blur sigmas
        weights: Weights for each scale
        normalize_method: Normalization method
    
    Returns:
        Enhanced image
    """
    # Apply MSR
    reflectance = multi_scale_retinex(img, sigmas, weights)
    
    # Convert back to RGB
    enhanced = log_to_rgb(reflectance)
    
    # Normalize
    enhanced = normalize_image(enhanced, method=normalize_method)
    
    return enhanced

def compare_ssr_msr(img, sigma_ssr=80, sigmas_msr=[15, 80, 250]):
    """
    Compare SSR and MSR results.
    
    Args:
        img: Input RGB image
        sigma_ssr: Sigma for SSR
        sigmas_msr: Sigmas for MSR
    
    Returns:
        Dictionary with original, SSR, and MSR results
    """
    # Run SSR
    enhanced_ssr = run_ssr(img, sigma_ssr)
    
    # Run MSR
    enhanced_msr = run_msr(img, sigmas_msr)
    
    return {
        'original': img,
        'ssr': enhanced_ssr,
        'msr': enhanced_msr
    } 