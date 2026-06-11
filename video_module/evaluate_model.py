import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Validation Dataset
val_dataset = datasets.ImageFolder(
    "deepfake_dataset/validation",
    transform=transform
)

# Test Dataset
test_dataset = datasets.ImageFolder(
    "deepfake_dataset/test",
    transform=transform
)

# DataLoaders
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Load ResNet18
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

# Evaluation Function
def evaluate(loader, name):

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"{name} Accuracy: {accuracy:.2f}%")

# Evaluate
evaluate(val_loader, "Validation")
evaluate(test_loader, "Test")