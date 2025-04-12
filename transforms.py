from PIL import Image
import numpy as np
import pywt
import cv2

# -------------------------------
# Haar Transform
# -------------------------------

def apply_haar_transform(image_path):
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    coeffs = pywt.dwt2(arr, 'haar')
    cA, (cH, cV, cD) = coeffs
    return cA, coeffs, img.size

def inverse_haar_transform(coeffs):
    reconstructed = pywt.idwt2(coeffs, 'haar')
    return Image.fromarray(np.clip(reconstructed, 0, 255).astype(np.uint8))

# -------------------------------
# Cosine Transform (DCT)
# -------------------------------

def apply_dct(image_path):
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    dct_result = cv2.dct(arr)
    return dct_result, arr.shape

def inverse_dct(dct_array, shape):
    idct_result = cv2.idct(dct_array)
    return Image.fromarray(np.clip(idct_result, 0, 255).astype(np.uint8))
