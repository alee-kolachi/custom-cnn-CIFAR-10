import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import io
from models.convnet import ConvNet

app = FastAPI()

MODEL_PATH = "best_convnet_cifar10.pth"

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
    
#Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Transform for test images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])


#Inference API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    images_bytes = await file.read()
    img = Image.open(io.BytesIO(images_bytes)).convert("RGB")

    img = transform(img).unsqueeze(0).to(device)

    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

    label = CLASSES[predicted.item()]
    return {"prediction": label}