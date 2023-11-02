import torch
from lightning import LightningModule
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from torchmetrics import F1Score, Accuracy, Precision, Recall
from transformers import ViTForImageClassification


class ViT(LightningModule):
    def __init__(self, data_module, pretrained_ckpt_name):
        super().__init__()
                
        self.model = ViTForImageClassification.from_pretrained(
            pretrained_ckpt_name,
            num_labels=data_module.num_classes,
            id2label=data_module.id2class,
            label2id=data_module.class2id
        )

        self.class_weights = torch.tensor(data_module.class_weights)

        self.train_f1 = F1Score(task='multiclass', average='macro', num_classes=data_module.num_classes)
        self.val_f1 = F1Score(task='multiclass', average='macro', num_classes=data_module.num_classes)
        self.test_f1 = F1Score(task='multiclass', average='macro', num_classes=data_module.num_classes)

        self.train_accuracy = Accuracy(task='multiclass', average='macro', num_classes=data_module.num_classes)
        self.val_accuracy = Accuracy(task='multiclass', average='macro', num_classes=data_module.num_classes)
        self.test_accuracy = Accuracy(task='multiclass', average='macro', num_classes=data_module.num_classes)

        self.train_precision = Precision(task='multiclass', average='macro', num_classes=data_module.num_classes)
        self.val_precision = Precision(task='multiclass', average='macro', num_classes=data_module.num_classes)
        self.test_precision = Precision(task='multiclass', average='macro', num_classes=data_module.num_classes)

        self.train_recall = Recall(task='multiclass', average='macro', num_classes=data_module.num_classes)
        self.val_recall = Recall(task='multiclass', average='macro', num_classes=data_module.num_classes)
        self.test_recall = Recall(task='multiclass', average='macro', num_classes=data_module.num_classes)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs.logits

    def common_step(self, batch):
        pixel_values, class_ids = batch.values()
        logits = self(pixel_values)
        loss = cross_entropy(
            input=logits,
            target=class_ids,
            weight=self.class_weights.to(self.device)
        )
        preds = logits.argmax(-1)
        return loss, preds, class_ids

    def training_step(self, batch, batch_idx):
        loss, preds, class_ids = self.common_step(batch)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        self.train_f1(preds, class_ids)
        self.log('train_f1', self.train_f1, prog_bar=False, on_step=False, on_epoch=True)

        self.train_accuracy(preds, class_ids)
        self.log('train_accuracy', self.train_accuracy, prog_bar=False, on_step=False, on_epoch=True)

        self.train_precision(preds, class_ids)
        self.log('train_precision', self.train_precision, prog_bar=False, on_step=False, on_epoch=True)

        self.train_recall(preds, class_ids)
        self.log('train_recall', self.train_recall, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, class_ids = self.common_step(batch)

        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        self.val_f1(preds, class_ids)
        self.log('val_f1', self.val_f1, prog_bar=False, on_step=False, on_epoch=True)

        self.val_accuracy(preds, class_ids)
        self.log('val_accuracy', self.val_accuracy, prog_bar=False, on_step=False, on_epoch=True)

        self.val_precision(preds, class_ids)
        self.log('val_precision', self.val_precision, prog_bar=False, on_step=False, on_epoch=True)

        self.val_recall(preds, class_ids)
        self.log('val_recall', self.val_recall, prog_bar=False, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, class_ids = self.common_step(batch)

        self.log('test_loss', loss, prog_bar=False, on_step=False, on_epoch=True)

        self.test_f1(preds, class_ids)
        self.log('test_f1', self.test_f1, prog_bar=False, on_step=False, on_epoch=True)

        self.test_accuracy(preds, class_ids)
        self.log('test_accuracy', self.test_accuracy, prog_bar=False, on_step=False, on_epoch=True)

        self.test_precision(preds, class_ids)
        self.log('test_precision', self.test_precision, prog_bar=False, on_step=False, on_epoch=True)

        self.test_recall(preds, class_ids)
        self.log('test_recall', self.test_recall, prog_bar=False, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        pass
