import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from mtcnn import MTCNN

# =========================
# VIDEO INPUT
# =========================

video_path = input("Enter video path: ")

# =========================
# FOLDERS
# =========================

frames_folder = "data/frames"
faces_folder = "data/faces"
processed_folder = "data/processed_faces"

# Create folders if needed
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(faces_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

# Clear old files
for folder in [frames_folder, faces_folder, processed_folder]:
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))

# =========================
# STEP 1: EXTRACT FRAMES
# =========================

video = cv2.VideoCapture(video_path)

count = 0

while True:

    success, frame = video.read()

    if not success:
        break

    frame_path = os.path.join(
        frames_folder,
        f"frame_{count}.jpg"
    )

    cv2.imwrite(frame_path, frame)

    count += 1

print(f"{count} frames extracted")

# =========================
# STEP 2: DETECT FACES
# =========================

detector = MTCNN()

face_count = 0

for filename in os.listdir(frames_folder):

    image_path = os.path.join(
        frames_folder,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    try:
         results = detector.detect_faces(image)
    except Exception:
          continue

    for result in results:

      x, y, w, h = result["box"]

    if w <= 0 or h <= 0:
        continue

    x = max(0, x)
    y = max(0, y)

    face = image[y:y+h, x:x+w]

    if face.size == 0:
        continue

    face_path = os.path.join(
        faces_folder,
        f"face_{face_count}.jpg"
    )

    cv2.imwrite(face_path, face)

    face_count += 1

print(f"{face_count} faces detected")

# =========================
# STEP 3: PREPROCESS
# =========================

for filename in os.listdir(faces_folder):

    image_path = os.path.join(
        faces_folder,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    resized = cv2.resize(image, (224, 224))

    output_path = os.path.join(
        processed_folder,
        filename
    )

    cv2.imwrite(output_path, resized)

print("Faces preprocessed")

# =========================
# STEP 4: LOAD MODEL
# =========================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "models/video_model_v1.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

# =========================
# STEP 5: PREDICT
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

real_count = 0
fake_count = 0

for filename in os.listdir(processed_folder):

    image_path = os.path.join(
        processed_folder,
        filename
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

    if prediction == 0:
        fake_count += 1
    else:
        real_count += 1

# =========================
# FINAL RESULT
# =========================

print("\n===== RESULTS =====")

print("Real Faces :", real_count)
print("Fake Faces :", fake_count)

if real_count + fake_count == 0:
    print("\nNo faces detected in video.")
    exit()
if fake_count > real_count:
    print("\nFINAL PREDICTION: FAKE VIDEO")
else:
    print("\nFINAL PREDICTION: REAL VIDEO")