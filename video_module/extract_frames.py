import cv2
import os

# Path to video
video_path = "data/raw_videos/sample1.mp4"

# Folder where frames will be saved
output_folder = "data/frames"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video
video = cv2.VideoCapture(video_path)

count = 0

while True:
    success, frame = video.read()

    # Stop if video ends
    if not success:
        break

    # Frame file name
    frame_path = f"{output_folder}/frame_{count}.jpg"

    # Save frame
    cv2.imwrite(frame_path, frame)

    count += 1

print(f"{count} frames extracted successfully!")