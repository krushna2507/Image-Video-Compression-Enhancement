from PIL import Image
import numpy as np
from collections import Counter, defaultdict
import heapq

# -------------------------------
# Run-Length Encoding (RLE)
# -------------------------------

def rle_encode(image_path):
    img = Image.open(image_path).convert("L")  # Grayscale
    data = np.array(img).flatten()

    encoded = []
    prev_pixel = data[0]
    count = 1
    for pixel in data[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoded.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1
    encoded.append((prev_pixel, count))
    return encoded, img.size

def rle_decode(encoded_data, image_size):
    decoded = []
    for val, count in encoded_data:
        decoded.extend([val] * count)
    decoded_img = np.array(decoded, dtype=np.uint8).reshape(image_size[1], image_size[0])
    return Image.fromarray(decoded_img)

# -------------------------------
# Huffman Encoding
# -------------------------------

class Node:
    def __init__(self, char=None, freq=0):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [Node(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(freq=node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)
    
    return heap[0]

def build_codes(root):
    codes = {}
    def generate_codes(node, current_code=""):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
    generate_codes(root)
    return codes

def huffman_encode(image_path):
    img = Image.open(image_path).convert("L")
    data = np.array(img).flatten()
    freq = Counter(data)

    tree = build_huffman_tree(freq)
    codes = build_codes(tree)
    
    encoded = ''.join([codes[pixel] for pixel in data])
    return encoded, codes, img.size

def huffman_decode(encoded, codes, image_size):
    reverse_codes = {v: k for k, v in codes.items()}
    current = ""
    decoded = []
    for bit in encoded:
        current += bit
        if current in reverse_codes:
            decoded.append(reverse_codes[current])
            current = ""
    decoded_img = np.array(decoded, dtype=np.uint8).reshape(image_size[1], image_size[0])
    return Image.fromarray(decoded_img)
