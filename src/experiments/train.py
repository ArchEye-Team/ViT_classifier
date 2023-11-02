import os
from pathlib import Path
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger

from transformers import ViTImageProcessor

from src.data.datamodule import DataModule
from src.model.vit import ViT


if __name__ == '__main__':
    # for connection to mlflow server
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'xxx'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'xxx'

    TRACKING_URI = 'xxx'
    EXPERIMENT_NAME = 'ViT'
    RUN_NAME = 'vit-base-patch16-224-in21k_on_dataset_v2'

    WEIGHTS_LABEL = 'google/vit-base-patch16-224-in21k'

    processor = ViTImageProcessor.from_pretrained(WEIGHTS_LABEL)
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    
    seed_everything(seed=42, workers=True)

    root_dir = Path('/kaggle/')

    data_module = DataModule(
        data_path=root_dir / 'input' / 'archeye-dataset' / 'ArchEyeDataset',
        batch_size=32,
        num_workers=4,
        image_size=size,
        mean_value=image_mean,
        std_value=image_std,
    )

    # load pretrained model
    model = ViT(data_module, WEIGHTS_LABEL)

    # train
    trainer = Trainer(
        default_root_dir=root_dir / 'working',
        deterministic=True,
        max_epochs=-1,
        accelerator='gpu',
        logger=MLFlowLogger(
            tracking_uri=TRACKING_URI,
            experiment_name=EXPERIMENT_NAME,
            run_name=RUN_NAME,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=root_dir / 'working' / 'checkpoints',
                save_last=True,
                save_top_k=3,
                monitor='val_f1',
                mode='max',
            ),
            EarlyStopping(
                patience=15,
                monitor='val_f1',
                mode='max',
            )
        ]
    )
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path='last'
    )

    # test it at the end of training
    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path='best'
    )
