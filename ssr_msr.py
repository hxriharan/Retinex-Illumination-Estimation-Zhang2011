import numpy as np
import cv2

def single_scale_retinex(img, sigma=30):
    """
    Apply Single Scale Retinex (SSR) to an RGB image.
    
    Parameters:
        img (np.ndarray): Input image (uint8, RGB).
        sigma (float): Standard deviation for Gaussian blur.
    
    Returns:
        np.ndarray: SSR-enhanced image (float32, normalized 0–1).
    """
    img = img.astype(np.float32) + 1.0  # Avoid log(0)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # SSR formula
    retinex = np.log(img) - np.log(blurred)
    
    # Normalize to [0, 1]
    retinex = normalize_01(retinex)
    return retinex


def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    """
    Apply Multi Scale Retinex (MSR) to an RGB image.
    
    Parameters:
        img (np.ndarray): Input image (uint8, RGB).
        sigmas (list): List of Gaussian sigmas to use.
    
    Returns:
        np.ndarray: MSR-enhanced image (float32, normalized 0–1).
    """
    img = img.astype(np.float32) + 1.0  # Avoid log(0)

    msr_result = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex = np.log(img) - np.log(blurred)
        msr_result += retinex
    
    msr_result /= len(sigmas)
    msr_result = normalize_01(msr_result)
    return msr_result


def normalize_01(img):
    """Normalize image to [0, 1] per channel."""
    img_min = np.min(img, axis=(0, 1), keepdims=True)
    img_max = np.max(img, axis=(0, 1), keepdims=True)
    return (img - img_min) / (img_max - img_min + 1e-6)
