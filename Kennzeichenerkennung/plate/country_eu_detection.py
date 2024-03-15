import torchvision
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

def eu_detection(img, model, transform):

    device = 'cpu'

    height, width = img.shape[:2]
    img = img[0:height, 0:width//4]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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