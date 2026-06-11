import cv2
import os

# Input folder with cropped faces
input_folder = "data/faces"

# Output folder for processed faces
output_folder = "data/processed_faces"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Loop through face images
for filename in os.listdir(input_folder):

    image_path = os.path.join(input_folder, filename)

    image = cv2.imread(image_path)

    # Skip invalid images
    if image is None:
        continue

    # Resize image to 224x224
    resized = cv2.resize(image, (224, 224))

    # Save processed image
    output_path = os.path.join(output_folder, filename)

    cv2.imwrite(output_path, resized)

print("All faces resized successfully!")