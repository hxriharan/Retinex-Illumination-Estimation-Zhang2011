import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from retinex.gradients import compute_gradients, visualize_gradients

def rgb_to_log(img, epsilon=1e-6):
    img = img.astype(np.float32) / 255.0
    return np.log(img + epsilon)

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0

def save_figure(fig, out_path):
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {out_path}")

def plot_gradients_per_channel(grad_x, grad_y, out_dir):
    channels = ['R', 'G', 'B']
    for i, ch in enumerate(channels):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].imshow(grad_x[:, :, i], cmap='gray')
        axes[0].set_title(f'Grad X - {ch}')
        axes[0].axis('off')

        axes[1].imshow(grad_y[:, :, i], cmap='gray')
        axes[1].set_title(f'Grad Y - {ch}')
        axes[1].axis('off')

        fig.tight_layout()
        save_figure(fig, os.path.join(out_dir, f'grad_xy_{ch}.png'))
        plt.close(fig)

def plot_quiver_field(image, grad_x, grad_y, out_dir):
    # Use grayscale image for background
    gray = to_grayscale(image)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gray, cmap='gray')
    ax.set_title('Gradient Vector Field (Quiver)')

    # Subsample for readability
    step = 20
    Y, X = np.mgrid[0:gray.shape[0]:step, 0:gray.shape[1]:step]
    U = grad_x[::step, ::step, 0]
    V = grad_y[::step, ::step, 0]

    ax.quiver(X, Y, U, -V, color='red', scale=1, scale_units='xy', angles='xy')
    ax.axis('off')

    save_figure(fig, os.path.join(out_dir, 'gradient_quiver_field.png'))
    plt.close(fig)

def main():
    image_path = 'Files/test.jpg'
    out_dir = 'Files/outputs'
    os.makedirs(out_dir, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save original
    fig = plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')
    save_figure(fig, os.path.join(out_dir, 'original_image.png'))
    plt.close(fig)

    # Log-RGB gradients
    log_image = rgb_to_log(image)
    grad_x_log, grad_y_log = compute_gradients(log_image, method='central')
    grad_mag_log = visualize_gradients(grad_x_log, grad_y_log)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(grad_mag_log, cmap='gray')
    axes[0].set_title('Log Gradient Magnitude')
    axes[0].axis('off')
    axes[1].imshow(grad_mag_log, cmap='inferno')
    axes[1].set_title('Zoom on Edges')
    axes[1].axis('off')
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, 'log_gradient_magnitude.png'))
    plt.close(fig)

    # Grayscale gradients
    gray_img = to_grayscale(image)
    gray_img_3ch = np.stack([gray_img]*3, axis=-1)
    grad_x_gray, grad_y_gray = compute_gradients(gray_img_3ch, method='central')
    grad_mag_gray = visualize_gradients(grad_x_gray, grad_y_gray)

    fig = plt.figure()
    plt.imshow(grad_mag_gray, cmap='gray')
    plt.title('Grayscale Gradient Magnitude')
    plt.axis('off')
    save_figure(fig, os.path.join(out_dir, 'grayscale_gradient_magnitude.png'))
    plt.close(fig)

    # Per-channel gradient visualizations
    plot_gradients_per_channel(grad_x_log, grad_y_log, out_dir)

    # Quiver field overlay
    plot_quiver_field(image, grad_x_log, grad_y_log, out_dir)

if __name__ == "__main__":
    main()
