import cv2
import os
import numpy as np
from moviepy.editor import VideoFileClip

# Extract frames from video
def extract_frames(video_path, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.png")
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    return count

# Simulated I/P/B frame labeling (just as visual demo)
def simulate_mpeg_labels(frame_count):
    labels = {}
    for i in range(frame_count):
        if i % 10 == 0:
            labels[i] = "I-frame"
        elif i % 5 == 0:
            labels[i] = "P-frame"
        else:
            labels[i] = "B-frame"
    return labels

# Apply point processing (e.g., brightness adjustment)
def enhance_frame(frame_path):
    frame = cv2.imread(frame_path)
    bright = cv2.convertScaleAbs(frame, alpha=1.1, beta=30)  # Increase brightness
    cv2.imwrite(frame_path.replace("frames", "enhanced"), bright)

# Reconstruct video from processed frames
def build_video_from_frames(folder, output_path, fps=20):
    images = sorted(os.listdir(folder))
    frame = cv2.imread(os.path.join(folder, images[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for img_name in images:
        img = cv2.imread(os.path.join(folder, img_name))
        out.write(img)
    out.release()

# Extract audio
def extract_audio(video_path, audio_path="audio.mp3"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)

# Combine audio with video
def add_audio_to_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = VideoFileClip(audio_path).audio
    final = video.set_audio(audio)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")
