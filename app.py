import streamlit as st
import os
from PIL import Image
import cv2

from image_compression import rle_encode, rle_decode, huffman_encode, huffman_decode
from lossy_compression import jpeg_compress
from transforms import apply_haar_transform, inverse_haar_transform, apply_dct, inverse_dct
from video_processing import (
    extract_frames, simulate_mpeg_labels, enhance_frame,
    build_video_from_frames, extract_audio, add_audio_to_video
)

# --- Helper Functions ---

# Function to display images
def display_image(image_path, caption="Image"):
    img = Image.open(image_path)
    st.image(img, caption=caption, use_column_width=True)

# Function to show a video thumbnail (first frame)
def display_video_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        st.image(img, caption="Video Thumbnail", use_column_width=True)
    cap.release()

# --- Streamlit UI ---

st.title("Image & Video Compression & Enhancement")

st.sidebar.header("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload an Image or Video", type=["png", "jpg", "jpeg", "mp4", "avi"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    st.sidebar.write(f"Uploaded File: {file_name}")
    
    # Save the uploaded file locally
    with open(file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_name.lower().endswith(('.mp4', '.avi')):
        st.sidebar.write("Video File Detected")
        display_video_thumbnail(file_name)
    else:
        st.sidebar.write("Image File Detected")
        display_image(file_name)

    # --- Image Compression Operations ---

    # RLE Compression
    if st.sidebar.button("Apply RLE Compression"):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            encoded, size = rle_encode(file_name)
            decoded_img = rle_decode(encoded, size)
            decoded_img_path = "rle_decoded.png"
            decoded_img.save(decoded_img_path)
            st.image(decoded_img_path, caption="RLE Decoded Image", use_column_width=True)
            st.success(f"RLE Compression applied! Encoded length: {len(encoded)}")
        else:
            st.warning("Please upload an image file for RLE.")

    # Huffman Compression
    if st.sidebar.button("Apply Huffman Compression"):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            encoded, codes, size = huffman_encode(file_name)
            decoded_img = huffman_decode(encoded, codes, size)
            decoded_img_path = "huffman_decoded.png"
            decoded_img.save(decoded_img_path)
            st.image(decoded_img_path, caption="Huffman Decoded Image", use_column_width=True)
            st.success(f"Huffman Compression applied! Encoded bits: {len(encoded)}")
        else:
            st.warning("Please upload an image file for Huffman.")

    # JPEG Compression
    if st.sidebar.button("Apply JPEG-like Compression"):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = jpeg_compress(file_name, quality=30)
            img.save("jpeg_compressed.png")
            st.image("jpeg_compressed.png", caption="JPEG Compressed Image", use_column_width=True)
            st.success("JPEG Compression applied!")
        else:
            st.warning("Please upload an image file for JPEG Compression.")

    # Haar Transform
    if st.sidebar.button("Apply Haar Transform"):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            cA, coeffs, _ = apply_haar_transform(file_name)
            img = inverse_haar_transform(coeffs)
            img.save("haar_reconstructed.png")
            st.image("haar_reconstructed.png", caption="Haar Reconstructed Image", use_column_width=True)
            st.success("Haar Transform applied!")
        else:
            st.warning("Please upload an image file for Haar Transform.")

    # DCT Transform
    if st.sidebar.button("Apply DCT Transform"):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            dct_array, shape = apply_dct(file_name)
            img = inverse_dct(dct_array, shape)
            img.save("dct_reconstructed.png")
            st.image("dct_reconstructed.png", caption="DCT Reconstructed Image", use_column_width=True)
            st.success("DCT Transform applied!")
        else:
            st.warning("Please upload an image file for DCT Transform.")

    # --- Video Processing Operations ---

    # Extract Frames
    if st.sidebar.button("Extract Frames from Video"):
        if file_name.lower().endswith(('.mp4', '.avi')):
            os.makedirs("frames", exist_ok=True)
            count = extract_frames(file_name, output_folder="frames")
            labels = simulate_mpeg_labels(count)
            st.success(f"Extracted {count} frames. First frame: {labels[0]}")
        else:
            st.warning("Please upload a video file for frame extraction.")

    # Enhance Frames
    if st.sidebar.button("Enhance Frames"):
        if os.path.exists("frames"):
            os.makedirs("enhanced", exist_ok=True)
            for f in os.listdir("frames"):
                enhance_frame(os.path.join("frames", f))
            st.success("Frames enhanced!")
        else:
            st.warning("No frames found. Please extract frames first.")

    # Rebuild Video
    if st.sidebar.button("Rebuild Video from Frames"):
        if os.path.exists("enhanced"):
            build_video_from_frames("enhanced", "enhanced_output.mp4")
            st.video("enhanced_output.mp4")
            st.success("Rebuilt video from enhanced frames!")
        else:
            st.warning("No enhanced frames found. Please enhance frames first.")

    # Extract Audio
    if st.sidebar.button("Extract Audio from Video"):
        if file_name.lower().endswith(('.mp4', '.avi')):
            extract_audio(file_name)
            st.success("Audio extracted successfully!")
        else:
            st.warning("Please upload a video file for audio extraction.")

    # Merge Audio and Video
    if st.sidebar.button("Merge Audio with Video"):
        if os.path.exists("enhanced_output.mp4") and os.path.exists("audio.mp3"):
            add_audio_to_video("enhanced_output.mp4", "audio.mp3", "final_output.mp4")
            st.video("final_output.mp4")
            st.success("Audio and video combined successfully!")
        else:
            st.warning("Ensure both enhanced video and audio are available.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Welcome to the Image and Video Compression & Enhancement App!")

