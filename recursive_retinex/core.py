import cv2
import numpy as np

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