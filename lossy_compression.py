from PIL import Image
import numpy as np
import cv2

# -------------------------------
# DCT-based JPEG-like Compression
# -------------------------------

def jpeg_compress(image_path, quality=50):
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    height, width = arr.shape
    dct_img = np.zeros_like(arr)

    # Apply DCT block-wise
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = arr[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                dct_block = cv2.dct(block)
                # Quantization (simple quality-based thresholding)
                dct_block[np.abs(dct_block) < quality] = 0
                dct_img[i:i+block_size, j:j+block_size] = dct_block

    # Inverse DCT to reconstruct
    idct_img = np.zeros_like(arr)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = dct_img[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                idct_img[i:i+block_size, j:j+block_size] = cv2.idct(block)

    final_img = np.clip(idct_img, 0, 255).astype(np.uint8)
    return Image.fromarray(final_img)
