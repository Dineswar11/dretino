from dretino import config

import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data import SwaVCollateFunction
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes


class SwaV(pl.LightningModule):
    def __init__(self, pretrained):
        super().__init__()
        if pretrained == "supervised":
            resnet = timm.create_model("resnet50", pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        else:
            from pl_bolts.models.self_supervised import SwAV

            weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/bolts_swav_imagenet/swav_imagenet.ckpt"
            swav = SwAV.load_from_checkpoint(weight_path, strict=False)

            resnet = swav.model
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection_head = SwaVProjectionHead(2048, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)

        self.criterion = SwaVLoss(sinkhorn_gather_distributed=True)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def training_step(self, batch, batch_idx):
        self.prototypes.normalize()
        crops, _, _ = batch
        multi_crop_features = [self.forward(x.to(self.device)) for x in crops]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim


model = SwaV()

dataset = LightlyDataset(input_dir="../aptos/train_images_resize/")

collate_fn = SwaVCollateFunction()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=config.NUM_WORKERS,
)

gpus = [4, 5, 6, 7]

trainer = pl.Trainer(
    max_epochs=150, gpus=gpus, strategy="ddp", sync_batchnorm=True
)
trainer.fit(model=model, train_dataloaders=dataloader)
