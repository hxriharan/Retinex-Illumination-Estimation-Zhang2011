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

def compute_gradients(img, method='central'):
    """
    Compute gradients using finite difference method.
    
    Args:
        img (H, W, C): Input image (log or grayscale stack)
        method (str): 'central', 'forward', or 'backward'
    
    Returns:
        grad_x, grad_y: (H, W, C) arrays for x and y gradients
    """
    img = img.astype(np.float32)
    H, W, C = img.shape
    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)

    for c in range(C):
        if method == 'central':
            grad_x[1:-1, :, c] = (img[2:, :, c] - img[:-2, :, c]) / 2.0
            grad_y[:, 1:-1, c] = (img[:, 2:, c] - img[:, :-2, c]) / 2.0
        elif method == 'forward':
            grad_x[:-1, :, c] = img[1:, :, c] - img[:-1, :, c]
            grad_y[:, :-1, c] = img[:, 1:, c] - img[:, :-1, c]
        elif method == 'backward':
            grad_x[1:, :, c] = img[1:, :, c] - img[:-1, :, c]
            grad_y[:, 1:, c] = img[:, 1:, c] - img[:, :-1, c]
        else:
            raise ValueError("Method must be 'central', 'forward', or 'backward'")

    return grad_x, grad_y

def estimate_illumination(img_log, method='gaussian', ksize=15, sigma=30):
    """
    Estimate illumination using Gaussian or bilateral filtering.
    
    Args:
        img_log: Image in log domain
        method: 'gaussian' or 'bilateral'
        ksize: Kernel size for Gaussian blur
        sigma: Sigma for Gaussian blur
    
    Returns:
        Estimated illumination in log domain
    """
    illum = np.zeros_like(img_log)
    for c in range(3):
        if method == 'gaussian':
            illum[:, :, c] = cv2.GaussianBlur(img_log[:, :, c], (ksize, ksize), sigma)
        elif method == 'bilateral':
            # Convert single channel to 8U before bilateral
            channel = img_log[:, :, c]
            norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            filtered = cv2.bilateralFilter(norm, d=31, sigmaColor=100, sigmaSpace=20)
            illum[:, :, c] = cv2.normalize(filtered.astype(np.float32), None, channel.min(), channel.max(), cv2.NORM_MINMAX)
        else:
            raise ValueError(f"Unsupported method: {method}")
    return illum

def recover_reflectance(log_img, log_illum):
    """Recover reflectance from log image and log illumination."""
    return log_img - log_illum

def recursive_filter_8dir(img, num_iterations=3, lambda_param=0.1):
    """
    Apply recursive filtering in 8 directions as per Zhang 2011.
    
    Args:
        img: Input image in log domain
        num_iterations: Number of recursive iterations
        lambda_param: Smoothing parameter
    
    Returns:
        Filtered image
    """
    # This is a placeholder for the 8-directional recursive filtering
    # Implementation would follow the specific algorithm from Zhang 2011
    filtered = img.copy()
    
    for _ in range(num_iterations):
        # Apply filtering in 8 directions
        # This is a simplified version - the actual implementation would be more complex
        for c in range(3):
            # Apply smoothing in 8 directions
            kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
            filtered[:, :, c] = cv2.filter2D(filtered[:, :, c], -1, kernel)
    
    return filtered

def sigmoid_tone_compression(img, alpha=1.0, beta=0.5):
    """
    Apply sigmoid tone compression to enhance image contrast.
    
    Args:
        img: Input image (0-255 range)
        alpha: Controls the steepness of the sigmoid curve
        beta: Controls the midpoint of the sigmoid curve
    
    Returns:
        Tone-compressed image
    """
    # Normalize to [0, 1]
    img_norm = img.astype(np.float32) / 255.0
    
    # Apply sigmoid transformation
    compressed = 1.0 / (1.0 + np.exp(-alpha * (img_norm - beta)))
    
    # Scale back to [0, 255]
    return (compressed * 255.0).astype(np.uint8)

def enhance_contrast(img, method='sigmoid', **kwargs):
    """
    Enhance image contrast using various methods.
    
    Args:
        img: Input image
        method: 'sigmoid', 'normalize', or 'both'
        **kwargs: Additional parameters for specific methods
    
    Returns:
        Enhanced image
    """
    if method == 'sigmoid':
        return sigmoid_tone_compression(img, **kwargs)
    
    elif method == 'normalize':
        return normalize_image(img, **kwargs)
    
    elif method == 'both':
        # Apply both sigmoid and normalization
        sigmoid_result = sigmoid_tone_compression(img, **kwargs)
        return normalize_image(sigmoid_result, **kwargs)
    
    else:
        raise ValueError("Method must be 'sigmoid', 'normalize', or 'both'")

def adjust_gamma(img, gamma=1.0):
    """
    Apply gamma correction to the image.
    
    Args:
        img: Input image
        gamma: Gamma value (gamma < 1 brightens, gamma > 1 darkens)
    
    Returns:
        Gamma-corrected image
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction
    return cv2.LUT(img, table)

def run_irr_pipeline(img, illum_method='gaussian', illum_ksize=15, illum_sigma=30,
                     use_recursive=False, num_iterations=3, lambda_param=0.1,
                     enhance_method='sigmoid', alpha=1.0, beta=0.5, gamma=1.0):
    """
    Run the complete Improved Recursive Retinex pipeline.
    
    Args:
        img: Input RGB image
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
    # Convert to log domain
    log_image = rgb_to_log(img)
    
    # Estimate illumination
    if use_recursive:
        log_illum = recursive_filter_8dir(log_image, num_iterations, lambda_param)
    else:
        log_illum = estimate_illumination(log_image, illum_method, illum_ksize, illum_sigma)
    
    # Recover reflectance
    log_reflect = recover_reflectance(log_image, log_illum)
    reflectance = log_to_rgb(log_reflect)
    illum_rgb = log_to_rgb(log_illum)
    
    # Enhance contrast
    enhanced = enhance_contrast(reflectance, method=enhance_method, alpha=alpha, beta=beta)
    
    # Apply gamma correction if needed
    if gamma != 1.0:
        enhanced = adjust_gamma(enhanced, gamma)
    
    return {
        'original': img,
        'illumination': illum_rgb,
        'reflectance': reflectance,
        'enhanced': enhanced
    } 