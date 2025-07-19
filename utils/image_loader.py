import numpy as np
import cv2

def load_hdr_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) + 1e-6  # avoid log(0)
    return np.log(img)
