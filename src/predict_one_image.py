import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from config import MODEL_DIR

def load_model(model_path=MODEL_DIR, num_classes=2):
    # Load the fine-tuned model
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path):
    # Define the transformation applied to the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the input image to a fixed size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensor
    ])

    # Load and preprocess the input image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_image(model, image_tensor, device):
    # Assuming you have a GPU available, you can move the model to the GPU
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_label = predicted.item()

    return predicted_label


def display_image(image_path):
    # Display the image
    image = Image.open(image_path)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    # Set the path to your input image
    image_path = r"../images/test1.jpg"


    # Load the model
    model = load_model()

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Make the prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_label = predict_image(model, image_tensor, device)

    # Display the predicted label and image
    if predicted_label == 0:
        print("The parking slot is empty.")
    else:
        print("The parking slot is occupied.")

    display_image(image_path)

