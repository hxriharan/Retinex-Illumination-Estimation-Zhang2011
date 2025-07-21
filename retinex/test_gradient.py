import cv2
import numpy as np
import matplotlib.pyplot as plt
from retinex.gradients import compute_gradients, visualize_gradients

def rgb_to_log(img, epsilon=1e-6):
    """
    Convert RGB image to log domain (log(R), log(G), log(B))
    Args:
        img (np.ndarray): RGB image in range [0, 255]
        epsilon (float): Small constant to avoid log(0)

    Returns:
        log_img (np.ndarray): Log-transformed image
    """
    img = img.astype(np.float32) / 255.0
    return np.log(img + epsilon)

def main():
    # Load an RGB image
    image_path = '/Users/hariharansureshkumar/Retinex_Project/Files/test.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to log space
    log_image = rgb_to_log(image)

    # Compute gradients
    grad_x, grad_y = compute_gradients(log_image)

    # Visualize
    grad_mag = visualize_gradients(grad_x, grad_y)

    # Display
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Log Gradient Magnitude")
    plt.imshow(grad_mag, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Zoom on Edges")
    plt.imshow(grad_mag, cmap='inferno')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
