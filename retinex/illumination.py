import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def rgb_to_log(img, epsilon=1e-6):
    img = img.astype(np.float32) / 255.0
    return np.log(img + epsilon)

def log_to_rgb(log_img):
    img = np.exp(log_img)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def estimate_illumination(img_log, method='gaussian', ksize=7, sigma=2):  # (ksize=7, sigma=2), (ksize=15, sigma=5), (ksize=31, sigma=15), (ksize=51, sigma=25) / original -> (15, 30)
    illum = np.zeros_like(img_log)
    for c in range(3):
        if method == 'gaussian':
            illum[:, :, c] = cv2.GaussianBlur(img_log[:, :, c], (ksize, ksize), sigma)
        elif method == 'bilateral':
            # Convert single channel to 8U before bilateral
            channel = img_log[:, :, c]
            norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            filtered = cv2.bilateralFilter(norm, d=9, sigmaColor=75, sigmaSpace=75)       # (d, sigmaColor, sigmaSpace) = (7, 25, 5), (15, 50, 10), (31, 100, 20)
            illum[:, :, c] = cv2.normalize(filtered.astype(np.float32), None, channel.min(), channel.max(), cv2.NORM_MINMAX)
        else:
            raise ValueError(f"Unsupported method: {method}")
    return illum

def recover_reflectance(log_img, log_illum):
    return log_img - log_illum

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main():
    # Load and preprocess image
    image_path = 'Files/test.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    log_image = rgb_to_log(image)

    # Try both Gaussian and Bilateral
    for method in ['gaussian', 'bilateral']:
        log_illum = estimate_illumination(log_image, method=method)
        log_reflect = recover_reflectance(log_image, log_illum)
        reflectance = log_to_rgb(log_reflect)
        illum_rgb = log_to_rgb(log_illum)

        # Display
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"Illumination ({method})")
        plt.imshow(illum_rgb)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Reflectance ({method})")
        plt.imshow(reflectance)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Save outputs
        save_image(reflectance, f'Files/reflectance_7_2{method}.png')
        save_image(illum_rgb, f'Files/illumination_7_2{method}.png')

if __name__ == "__main__":
    main()
