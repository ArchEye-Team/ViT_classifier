import os
import torch
from PIL import Image
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize,
                                    Resize, 
                                    ToTensor)
import numpy as np


def load_model(weights_path: str):
    if not os.path.exists(weights_path):
        raise ValueError(f'Path to model weights is not valid: {weights_path}')

    model = torch.load(weights_path)
    return model


def preprocess_image(
    img_path: str, model_img_size: int = 224,
    mean_value=[0.5, 0.5, 0.5], std_value=[0.5, 0.5, 0.5],
):
    if not os.path.exists(img_path):
        raise ValueError(f'Path to input image is not valid: {img_path}')
    
    image = Image.open(img_path).convert('RGB')

    transforms = [
        Resize(model_img_size),
        CenterCrop(model_img_size),
        ToTensor(),
        Normalize(mean=mean_value, std=std_value),
    ]

    processed = Compose(transforms)(image)

    return processed[None, :, :, :]


def infer_loaded_model(model, processed_image):
    with torch.no_grad():
        logits = model.forward(processed_image).logits
    
    prediction = logits.argmax(-1)
    class_id = prediction.item()
    class_label = data_module.id2class[prediction.item()]
    
    probs = np.exp(logits[0]) / sum(np.exp(logits[0]))
    conf = probs[class_id].item()

    return class_id, class_label, conf


if __name__ == '__main__':
    TORCH_WEIGHTS_PATH = '/kaggle/working/vit.pt'
    IMG_PATH = '/kaggle/input/archeye-dataset/ArchEyeDataset/Барокко/100000.jpg'

    loaded_model = load_model(TORCH_WEIGHTS_PATH)

    processed_image = preprocess_image(IMG_PATH)
    class_id, class_label, confidence = infer_loaded_model(
        loaded_model, preprocess_image,
    )

    print(f'Predicted class on image is {class_label}, model confidence: {confidence:.2f}')
