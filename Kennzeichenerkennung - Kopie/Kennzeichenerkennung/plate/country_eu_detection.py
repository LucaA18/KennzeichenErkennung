import torchvision
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

def eu_detection(img):
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
    device = 'cpu'
    model = torchvision.models.mobilenet_v2(weights=weights).to(device)
    
    # Get the length of class_names (one output unit for each class)
    output_shape = 11

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)
    model.load_state_dict(torch.load('plate/licence_class.pth', map_location=torch.device('cpu')))
    transform = transforms.Compose([
    transforms.Resize(232),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
    ])
    height, width = img.shape[:2]
    img = img[0:height, 0:width//4]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konvertiere BGR zu RGB
    # Konvertiere das Numpy-Array in ein PIL-Image
    img = Image.fromarray(img)

    img = transform(img).to(device)

    model.eval()
    with torch.inference_mode():
        output = model(img.unsqueeze(0))

    class_map = {
        0: 'B',
        1: 'BG',
        2: 'D',
        3: 'E',
        4: 'F',
        5: 'GR',
        6: 'HR',
        7: 'I',
        8: 'NL',
        9: 'P',
        10: 'PL'
    }
    output_idx = np.argmax(output.cpu()).item()

    return class_map[output_idx]