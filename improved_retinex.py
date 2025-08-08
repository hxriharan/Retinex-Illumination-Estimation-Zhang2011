# irr.py
import cv2
import numpy as np

# ---------- basic helpers ----------
def rgb_to_log(img, epsilon=1e-6):
    """Convert RGB uint8 image to log-domain float32."""
    img = img.astype(np.float32) / 255.0
    return np.log(img + epsilon)

def log_to_rgb(log_img):
    """Convert log-domain image back to uint8 RGB."""
    img = np.exp(log_img)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def normalize_image(img, vmin=None, vmax=None):
    """Min–max normalize to uint8 for quick viewing."""
    img = img.astype(np.float32)
    vmin = float(img.min()) if vmin is None else vmin
    vmax = float(img.max()) if vmax is None else vmax
    if vmax - vmin < 1e-8:
        return np.full_like(img, 128, dtype=np.uint8)
    out = (img - vmin) / (vmax - vmin + 1e-12)
    return (out * 255.0).astype(np.uint8)

# ---------- improved recursive retinex ----------
def _scan_order(H, W, dy, dx):
    """Ensure predecessor (y-dy, x-dx) is visited first."""
    ys = range(H) if dy >= 0 else range(H - 1, -1, -1)
    xs = range(W) if dx >= 0 else range(W - 1, -1, -1)
    return ys, xs

def directional_recursive_filter(img_channel, direction, alpha=0.1, threshold=0.2):
    """
    Edge-aware one-pass recursion along a direction (log-domain).
    img_channel: (H,W) float32 log intensity
    direction: (dy, dx) in {-1,0,1}^2 \ {(0,0)}
    alpha: smoothing factor
    threshold: stop when |I(p)-I(prev)| >= threshold (log units)
    """
    H, W = img_channel.shape
    dy, dx = direction
    L = np.zeros_like(img_channel, dtype=np.float32)

    ys, xs = _scan_order(H, W, dy, dx)
    for y in ys:
        for x in xs:
            py, px = y - dy, x - dx
            if 0 <= py < H and 0 <= px < W:
                if abs(img_channel[y, x] - img_channel[py, px]) < threshold:
                    L[y, x] = (1 - alpha) * L[py, px] + alpha * img_channel[y, x]
                else:
                    L[y, x] = img_channel[y, x]  # restart at edges
            else:
                L[y, x] = img_channel[y, x]      # seed boundary
    return L

def improved_recursive_retinex(img_log, alpha=0.05, threshold=0.4,
                               use_auto_threshold=False, post_smooth=True):
    """
    8-direction symmetric recursive illumination (log-domain).
    - use_auto_threshold: compute channel-wise threshold from gradient stats
    - post_smooth: tiny Gaussian blur to hide directional seams
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    H, W, C = img_log.shape
    illum = np.zeros((H, W, C), dtype=np.float32)

    def _auto_threshold_from_grad(ch, k=4.0):
        gy = np.diff(ch, axis=0, prepend=ch[:1])
        gx = np.diff(ch, axis=1, prepend=ch[:, :1])
        g = np.abs(gx) + np.abs(gy)
        med = np.median(g)
        mad = np.median(np.abs(g - med)) + 1e-6
        return k * (med + 1.4826 * mad)

    for c in range(C):
        ch = img_log[:, :, c].astype(np.float32)
        th = _auto_threshold_from_grad(ch) if use_auto_threshold else threshold

        acc = np.zeros((H, W), dtype=np.float32)
        for dy, dx in directions:
            fwd = directional_recursive_filter(ch, (dy, dx), alpha=alpha, threshold=th)
            bwd = directional_recursive_filter(ch, (-dy, -dx), alpha=alpha, threshold=th)
            acc += 0.5 * (fwd + bwd)

        Lc = acc / float(len(directions))
        if post_smooth:
            Lc = cv2.GaussianBlur(Lc, (0, 0), 1.0)  # σ≈1
        illum[:, :, c] = Lc
    return illum


# ---------- reflectance + simple enhancement ----------
def recover_reflectance(log_img, log_illum):
    """Log reflectance (clamped to avoid numeric junk)."""
    log_R = log_img - log_illum
    return np.clip(log_R, -10.0, 10.0)

def enhance_reflectance_u8(R_u8, method='normalize'):
    """
    Simple visualization options:
      - 'normalize'    : min–max per-image
      - 'adaptive_hist': CLAHE on Y channel
      - 'gamma'        : auto gamma based on mean
    """
    if method == 'normalize':
        return normalize_image(R_u8)

    if method == 'adaptive_hist':
        yuv = cv2.cvtColor(R_u8, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    if method == 'gamma':
        m = max(1e-6, np.mean(R_u8) / 255.0)
        gamma = float(np.clip(-0.3 / np.log10(m + 1e-6), 0.5, 2.5))
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255.0 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(R_u8, table)

    return normalize_image(R_u8)

# ---------- IRR pipeline (recursive only) ----------
def run_irr_pipeline(img_u8, alpha=0.05, threshold=0.4, enhance_method='normalize',
                     use_auto_threshold=False, post_smooth=True):
    """
    Full IRR with consistent defaults and optional auto-threshold/post-smooth.
    """
    log_img = rgb_to_log(img_u8)
    log_L = improved_recursive_retinex(log_img, alpha=alpha, threshold=threshold,
                                       use_auto_threshold=use_auto_threshold,
                                       post_smooth=post_smooth)
    log_R = recover_reflectance(log_img, log_L)

    R_u8 = log_to_rgb(log_R)
    L_u8 = log_to_rgb(log_L)
    R_vis = enhance_reflectance_u8(R_u8, method=enhance_method)

    return {'original': img_u8, 'illumination': L_u8, 'reflectance': R_u8, 'enhanced': R_vis}


# ---------- quick self-test ----------
if __name__ == "__main__":
    # synthetic: black bg + white square
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[125:175, 125:175] = 255

    out = run_irr_pipeline(img, alpha=0.05, threshold=0.4, enhance_method='normalize')
    cv2.imwrite("irr_illum.png", cv2.cvtColor(out['illumination'], cv2.COLOR_RGB2BGR))
    cv2.imwrite("irr_reflect.png", cv2.cvtColor(out['reflectance'], cv2.COLOR_RGB2BGR))
    cv2.imwrite("irr_enhanced.png", cv2.cvtColor(out['enhanced'], cv2.COLOR_RGB2BGR))
