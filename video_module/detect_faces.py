from mtcnn import MTCNN
import cv2
import os

# Initialize face detector
detector = MTCNN()

# Input folder containing frames
input_folder = "data/frames"

# Output folder for faces
output_folder = "data/faces"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Loop through all frames
for filename in os.listdir(input_folder):

    image_path = os.path.join(input_folder, filename)

    image = cv2.imread(image_path)

    # Skip if image not loaded
    if image is None:
        continue

    # Detect faces
    results = detector.detect_faces(image)

    # Loop through detected faces
    for i, result in enumerate(results):

        x, y, width, height = result['box']

        # Crop face
        face = image[y:y+height, x:x+width]

        # Save face
        face_path = os.path.join(
            output_folder,
            f"face_{filename}"
        )

        cv2.imwrite(face_path, face)

print("Faces extracted successfully!")