import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from config import TRAINING_DATA_DIR, MODEL_DIR


class ParkingClassifier:
    def __init__(self, TRAINING_DATA_DIR, num_classes=2, train_ratio=0.8, batch_size=32, num_epochs=10):
        self.data_dir = TRAINING_DATA_DIR
        self.num_classes = num_classes
        self.train_ratio = train_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Assuming you have a GPU available, you can move the model and data loaders to the GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Learning using- {self.device}')

        # Define the transformation applied to each image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._load_datasets()
        self._initialize_model()

    def _load_datasets(self):
        # Load the dataset using ImageFolder
        dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create data loaders for training and validation sets
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)

    def _initialize_model(self):

        # Load a pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)

        # Freeze the weights of the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the output layer with a new fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        self.model = self.model.to(self.device)

    def train(self, learning_rate=0.001, momentum=0.9):
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()  # Make criterion an instance variable
        optimizer = optim.SGD(self.model.fc.parameters(), lr=learning_rate, momentum=momentum)
        self.model.train()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)  # Access criterion from instance variable

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Training Loss: {epoch_loss:.4f}")

            # Validation loop
            self._validate(epoch + 1, self.num_epochs)

    def _validate(self, current_epoch, total_epochs):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)  # Access criterion from instance variable
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(self.val_dataloader)
        val_accuracy = correct / total
        print(
            f"Epoch {current_epoch}/{total_epochs} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)


if __name__ == '__main__':
    parking_classifier = ParkingClassifier(TRAINING_DATA_DIR, num_epochs=10)
    parking_classifier.train()
    parking_classifier.save_model(MODEL_DIR)
