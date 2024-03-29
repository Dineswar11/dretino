from dretino import config

import torch
from torch import nn
import torchvision
import pytorch_lightning as pl

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(pl.LightningModule):
    def __init__(self, pretrained):
        super().__init__()
        if pretrained == "supervised":
            resnet = timm.create_model("resnet50", pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        else:
            from pl_bolts.models.self_supervised import SimCLR

            weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

            self.backbone = nn.Sequential(*list(simclr.encoder.children())[:-1])
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)

        # enable gather_distributed to gather features from all gpus
        # before calculating the loss
        self.criterion = NTXentLoss(gather_distributed=True)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim



model = SimCLR()

dataset = LightlyDataset(input_dir="../aptos/train_images_resize/")
# or create a dataset from a folder containing images or videos:
# dataset = LightlyDataset("path/to/folder")

collate_fn = SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.0,
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=config.NUM_WORKERS,
)

gpus = torch.cuda.device_count()

# train with DDP and use Synchronized Batch Norm for a more accurate batch norm
# calculation
trainer = pl.Trainer(
    max_epochs=250, gpus=gpus, strategy="ddp", sync_batchnorm=True
)
trainer.fit(model=model, train_dataloaders=dataloader)
