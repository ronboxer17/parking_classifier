import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# Define the number of classes for your parking slot classification task
num_classes = 2

# Set the path to your fine-tuned model
model_path = 'parking_classification_model.pth'

# Load the fine-tuned model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the transformation applied to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the input image to a fixed size
    transforms.ToTensor(),           # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensor
])

# Set the path to your input image
image_path = r"C:\Users\ronbo\Desktop\test6.jpg"

# Load and preprocess the input image
image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Assuming you have a GPU available, you can move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
image_tensor = image_tensor.to(device)

# Make the prediction
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    predicted_label = predicted.item()

# Display the predicted label and image
if predicted_label == 0:
    print("The parking slot is empty.")
else:
    print("The parking slot is occupied.")

plt.imshow(image)
plt.show() # image will not be displayed without this
