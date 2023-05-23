import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

# Define the number of classes for your parking slot classification task
num_classes = 2

# Set the path to your dataset directory
data_dir = r"C:\Users\ronbo\desktop\master\Courses\Third_year\Unsuperised Learning\parking\parking_classificar\data"

# Define the transformation applied to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the input images to a fixed size
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensors
])

# Load the dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and validation sets
train_ratio = 0.8  # 80% for training, 20% for validation
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Freeze the weights of the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Replace the output layer with a new fully connected layer
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Assuming you have a GPU available, you can move the model and data loaders to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# train_dataloader = train_dataloader.to(device)
# val_dataloader = val_dataloader.to(device)

# Training loop
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_dataloader)
    val_accuracy = correct / total

    print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

# Once training is complete, you can save the model
torch.save(model.state_dict(), 'parking_classification_model.pth')
