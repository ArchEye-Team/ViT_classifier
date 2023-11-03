import torch
from transformers import ViTImageProcessor

from src.data.datamodule import DataModule
from src.model.vit import ViT


if __name__ == '__main__':
    SAVED_CKPT_PATH = '/kaggle/input/vit-ckpt/epoch22-step3956.ckpt'
    DATASET_PATH = '/kaggle/input/archeye-dataset/ArchEyeDataset'
    WEIGHTS_LABEL = 'google/vit-base-patch16-224-in21k'
    OUTPUT_WEIGHTS = 'vit.pt'

    processor = ViTImageProcessor.from_pretrained(WEIGHTS_LABEL)
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]

    data_module = DataModule(
        data_path=DATASET_PATH,
        batch_size=32,
        num_workers=4,
        image_size=size,
        mean_value=image_mean,
        std_value=image_std,
    )

    lightning_model = ViT.load_from_checkpoint(
        SAVED_CKPT_PATH,
        data_module=data_module,
        pretrained_ckpt_name=WEIGHTS_LABEL,
    )

    torch_model = lightning_model.model
    torch.save(torch_model, OUTPUT_WEIGHTS)
