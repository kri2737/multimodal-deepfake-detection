import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
    torch.load(
        "models/video_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["fake", "real"]


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    return classes[prediction]


if __name__ == "__main__":
    # Test code only
    image_path = "deepfake_dataset/test/real/068_10.png"
    result = predict_image(image_path)
    print("Prediction:", result)