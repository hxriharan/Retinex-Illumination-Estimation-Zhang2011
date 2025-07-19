import numpy as np
import cv2

def compute_gradients(log_image):
    """
    Compute gradients ∂L/∂x and ∂L/∂y of the log image using Sobel filters.

    Args:
        log_image (np.ndarray): Log domain image (H, W, C)

    Returns:
        grad_x (np.ndarray): Horizontal gradients (H, W, C)
        grad_y (np.ndarray): Vertical gradients (H, W, C)
    """
    grad_x = np.zeros_like(log_image)
    grad_y = np.zeros_like(log_image)

    for c in range(log_image.shape[2]):
        grad_x[..., c] = cv2.Sobel(log_image[..., c], cv2.CV_64F, dx=1, dy=0, ksize=3)
        grad_y[..., c] = cv2.Sobel(log_image[..., c], cv2.CV_64F, dx=0, dy=1, ksize=3)

    return grad_x, grad_y


def visualize_gradients(grad_x, grad_y):
    """
    Create a visualization of gradient magnitudes for sanity check.

    Args:
        grad_x (np.ndarray): Gradient in x direction
        grad_y (np.ndarray): Gradient in y direction

    Returns:
        grad_magnitude (np.ndarray): Combined gradient magnitude image
    """
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # Average across channels for visualization
    if magnitude.ndim == 3:
        magnitude = np.mean(magnitude, axis=2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)
