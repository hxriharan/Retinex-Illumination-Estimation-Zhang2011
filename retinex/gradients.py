import numpy as np

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

def visualize_gradients(grad_x, grad_y):
    mag = np.sqrt(np.sum(grad_x**2 + grad_y**2, axis=2))
    mag /= np.max(mag) + 1e-8
    return mag
