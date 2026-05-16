import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Decide device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset

train_dataset = datasets.ImageFolder(
    "data/train",
    transform=train_transform
)

# Create data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_dataset = datasets.ImageFolder(
    "data/val",
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

# Load pretrained MobileNetV3
model = models.mobilenet_v3_small(pretrained=True)

# Replace final classification layer
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    3
)

model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

# Training loop
for epoch in range(epochs):

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Validation Accuracy: {accuracy:.2f}%")

    model.train()

# Save model
torch.save(model.state_dict(), "model/best_model.pth")

print("Training Finished")