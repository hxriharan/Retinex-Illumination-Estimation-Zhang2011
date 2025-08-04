import cv2
import numpy as np
import os

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

def normalize_image(img, method='minmax'):
    """
    Normalize image using different methods.
    
    Args:
        img: Input image
        method: 'minmax', 'histogram', or 'adaptive'
    
    Returns:
        Normalized image
    """
    if method == 'minmax':
        # Min-max normalization
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            normalized = (img - img_min) / (img_max - img_min)
            return (normalized * 255.0).astype(np.uint8)
        else:
            return img
    
    elif method == 'histogram':
        # Histogram equalization
        if len(img.shape) == 3:
            # For color images, apply to each channel
            normalized = np.zeros_like(img)
            for c in range(3):
                normalized[:, :, c] = cv2.equalizeHist(img[:, :, c])
            return normalized
        else:
            return cv2.equalizeHist(img)
    
    elif method == 'adaptive':
        # Adaptive histogram equalization
        if len(img.shape) == 3:
            # For color images, apply to each channel
            normalized = np.zeros_like(img)
            for c in range(3):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                normalized[:, :, c] = clahe.apply(img[:, :, c])
            return normalized
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img)
    
    else:
        raise ValueError("Method must be 'minmax', 'histogram', or 'adaptive'")

def color_correction(img, method='white_balance'):
    """
    Apply color correction to the image.
    
    Args:
        img: Input image
        method: 'white_balance', 'gamma', or 'histogram'
    
    Returns:
        Color-corrected image
    """
    if method == 'white_balance':
        # Simple white balance using gray world assumption
        result = img.copy().astype(np.float32)
        
        # Calculate mean for each channel
        means = np.mean(result, axis=(0, 1))
        
        # Find the channel with maximum mean
        max_mean = np.max(means)
        
        # Scale all channels to match the maximum
        for c in range(3):
            if means[c] > 0:
                result[:, :, c] *= max_mean / means[c]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    elif method == 'gamma':
        # Gamma correction
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    elif method == 'histogram':
        # Histogram equalization for color correction
        return normalize_image(img, method='histogram')
    
    else:
        raise ValueError("Method must be 'white_balance', 'gamma', or 'histogram'")

def resize_image(img, target_size=None, scale_factor=None):
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        img: Input image
        target_size: Target (width, height) or None
        scale_factor: Scale factor or None
    
    Returns:
        Resized image
    """
    if target_size is not None:
        # Resize to target size
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    elif scale_factor is not None:
        # Resize by scale factor
        height, width = img.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    else:
        return img

def create_comparison_image(images, labels=None, layout='horizontal'):
    """
    Create a comparison image from multiple images.
    
    Args:
        images: List of images
        labels: List of labels for each image
        layout: 'horizontal' or 'vertical'
    
    Returns:
        Comparison image
    """
    if not images:
        return None
    
    # Ensure all images have the same size
    target_size = (images[0].shape[1], images[0].shape[0])
    resized_images = [resize_image(img, target_size) for img in images]
    
    if layout == 'horizontal':
        # Stack horizontally
        comparison = np.hstack(resized_images)
    else:
        # Stack vertically
        comparison = np.vstack(resized_images)
    
    return comparison

def calculate_metrics(img1, img2):
    """
    Calculate image quality metrics between two images.
    
    Args:
        img1: Reference image
        img2: Test image
    
    Returns:
        Dictionary of metrics
    """
    # Convert to grayscale for some metrics
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = img1
    
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        gray2 = img2
    
    # Ensure same size
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    
    # Calculate metrics
    metrics = {}
    
    # Mean Squared Error (MSE)
    mse = np.mean((gray1.astype(np.float32) - gray2.astype(np.float32)) ** 2)
    metrics['mse'] = mse
    
    # Peak Signal-to-Noise Ratio (PSNR)
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        metrics['psnr'] = psnr
    else:
        metrics['psnr'] = float('inf')
    
    # Structural Similarity Index (SSIM) - simplified
    # Note: This is a simplified SSIM calculation
    mu1 = np.mean(gray1)
    mu2 = np.mean(gray2)
    sigma1 = np.var(gray1)
    sigma2 = np.var(gray2)
    sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    metrics['ssim'] = ssim
    
    return metrics 