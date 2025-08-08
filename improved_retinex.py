import cv2
import numpy as np


def rgb_to_log(img, epsilon=1e-6):
    img = img.astype(np.float32) / 255.0
    return np.log(img + epsilon)


def log_to_rgb(log_img):
    img = np.exp(log_img)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def normalize_image(img, vmin=None, vmax=None):
    img = img.astype(np.float32)
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()
    if vmax - vmin < 1e-6:  # Prevent division by zero
        return np.full_like(img, 128, dtype=np.uint8)
    norm = (img - vmin) / (vmax - vmin)
    return (norm * 255).astype(np.uint8)


def adaptive_recursive_filter(img_channel, direction, alpha=0.05):
    """
    Simplified recursive filter that preserves edges better
    """
    H, W = img_channel.shape
    output = img_channel.copy().astype(np.float32)
    dy, dx = direction
    
    # Determine processing order
    if dy >= 0:
        y_range = range(H)
    else:
        y_range = range(H-1, -1, -1)
        
    if dx >= 0:
        x_range = range(W)
    else:
        x_range = range(W-1, -1, -1)
    
    for y in y_range:
        for x in x_range:
            py, px = y - dy, x - dx
            
            if 0 <= py < H and 0 <= px < W:
                # Adaptive threshold based on local statistics
                local_std = np.std(img_channel[max(0, y-2):min(H, y+3), 
                                             max(0, x-2):min(W, x+3)])
                threshold = max(0.01, local_std * 0.1)
                
                diff = abs(img_channel[y, x] - output[py, px])
                
                if diff < threshold:
                    # Use smaller alpha for better edge preservation
                    local_alpha = alpha * (1.0 - diff / threshold)
                    output[y, x] = (1 - local_alpha) * output[py, px] + local_alpha * img_channel[y, x]
    
    return output


def improved_recursive_retinex(img_log, alpha=0.03, max_iterations=3):
    """
    Completely rewritten IRR algorithm with better edge handling
    """
    H, W, C = img_log.shape
    
    # Initialize illumination as the image itself
    illum = img_log.copy()
    
    directions = [(0, 1), 
                (1, 0), 
                (0, -1), 
                (-1, 0),  # 4-connected
                (1, 1),
                (1, -1), 
                (-1, 1), 
                (-1, -1)]  # 8-connected
    
    # Apply multiple iterations with decreasing effect
    for iteration in range(max_iterations):
        current_alpha = alpha * (0.5 ** iteration)  # Decreasing alpha
        temp_illum = np.zeros_like(illum)
        
        for c in range(C):
            channel_sum = np.zeros((H, W), dtype=np.float32)
            
            for direction in directions:
                filtered = adaptive_recursive_filter(illum[:, :, c], direction, current_alpha)
                channel_sum += filtered
                
            temp_illum[:, :, c] = channel_sum / len(directions)
        
        # Blend with previous iteration to prevent over-smoothing
        blend_factor = 0.7
        illum = blend_factor * temp_illum + (1 - blend_factor) * illum
    
    return illum


def estimate_illumination(img_log, method='gaussian', ksize=15, sigma=30):
    """Standard illumination estimation methods"""
    illum = np.zeros_like(img_log)
    for c in range(3):
        if method == 'gaussian':
            illum[:, :, c] = cv2.GaussianBlur(img_log[:, :, c], (ksize, ksize), sigma)
        elif method == 'bilateral':
            channel = img_log[:, :, c]
            # Convert to uint8 for bilateral filter
            norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            filtered = cv2.bilateralFilter(norm, d=31, sigmaColor=100, sigmaSpace=20)
            # Convert back to original range
            illum[:, :, c] = cv2.normalize(filtered.astype(np.float32), None, 
                                         channel.min(), channel.max(), cv2.NORM_MINMAX)
        else:
            raise ValueError(f"Unsupported method: {method}")
    return illum


def recover_reflectance(log_img, log_illum):
    """Recover reflectance with clamping to prevent artifacts"""
    reflectance = log_img - log_illum
    # Clamp extreme values to prevent artifacts
    reflectance = np.clip(reflectance, -10, 10)
    return reflectance


def enhanced_contrast_adjustment(img, method='adaptive_hist'):
    """
    Better contrast enhancement for high dynamic range images
    """
    if method == 'adaptive_hist':
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(img.shape) == 3:
            # Convert to YUV and apply CLAHE only to Y channel
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            yuv[:,:,0] = clahe.apply(yuv[:,:,0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img)
        return enhanced
    elif method == 'normalize':
        return normalize_image(img)
    elif method == 'gamma':
        # Adaptive gamma correction
        mean_val = np.mean(img) / 255.0
        gamma = -0.3 / np.log10(mean_val + 1e-6)
        gamma = np.clip(gamma, 0.5, 2.5)
        table = np.array([((i / 255.0) ** (1/gamma)) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img, table)
    else:
        return normalize_image(img)


def run_fixed_irr_pipeline(img, illum_method='gaussian', illum_ksize=15, illum_sigma=30,
                          use_recursive=False, alpha=0.03, max_iterations=3,
                          enhance_method='adaptive_hist'):
    """
    Completely rewritten IRR pipeline with proper edge preservation
    """
    # Convert to log domain
    log_image = rgb_to_log(img)

    # Estimate illumination using the chosen method
    if use_recursive:
        log_illum = improved_recursive_retinex(log_image, alpha=alpha, max_iterations=max_iterations)
    else:
        log_illum = estimate_illumination(log_image, method=illum_method, 
                                        ksize=illum_ksize, sigma=illum_sigma)

    # Recover reflectance
    log_reflect = recover_reflectance(log_image, log_illum)
    
    # Convert back to RGB domain
    reflectance = log_to_rgb(log_reflect)
    illum_rgb = log_to_rgb(log_illum)

    # Enhanced contrast adjustment
    enhanced = enhanced_contrast_adjustment(reflectance, method=enhance_method)

    return {
        'original': img,
        'illumination': illum_rgb,
        'reflectance': reflectance,
        'enhanced': enhanced
    }


def create_test_image():
    """Create a test image similar to your input"""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[125:175, 125:175] = 255  # White square in center
    return img


def test_all_methods():
    """Test function to compare all three methods"""
    img = create_test_image()
    
    print("Testing SSR...")
    log_img = rgb_to_log(img)
    log_illum_ssr = estimate_illumination(log_img, method='gaussian', ksize=15, sigma=80)
    ssr_result = normalize_image(log_to_rgb(recover_reflectance(log_img, log_illum_ssr)))
    
    print("Testing MSR...")
    scales = [15, 80, 250]
    msr_result = np.zeros_like(log_img)
    for sigma in scales:
        log_illum = estimate_illumination(log_img, method='gaussian', ksize=15, sigma=sigma)
        msr_result += recover_reflectance(log_img, log_illum)
    msr_result /= len(scales)
    msr_result = normalize_image(log_to_rgb(msr_result))
    
    print("Testing Fixed IRR...")
    irr_results = run_fixed_irr_pipeline(img, use_recursive=True, alpha=0.03, 
                                       max_iterations=2, enhance_method='normalize')
    
    return {
        'original': img,
        'ssr': ssr_result,
        'msr': msr_result,
        'irr': irr_results['enhanced']
    }

# Usage example:
# results = test_all_methods()
# The IRR result should now preserve contrast instead of being uniform grey