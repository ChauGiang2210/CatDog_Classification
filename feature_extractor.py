import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = torch.load('train/resnet50_feature_extractor.pth', map_location=device, weights_only=False)
resnet.to(device)

#transform ảnh đầu vào
Transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def Feature_Extraction_img(image):
    features=[]
    # file_name = os.path.basename(img_path)
    # if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')): return
    # img = Image.open(img_path)
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)
    img = Transforms(img)
    batch = torch.unsqueeze(img, 0).to(device)
    with torch.no_grad():
      fc7_features = resnet(batch)
      features.append(fc7_features.squeeze().cpu())
    return features