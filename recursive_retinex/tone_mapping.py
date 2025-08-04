import numpy as np
import cv2

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