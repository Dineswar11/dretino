import copy

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import MSNLoss
from lightly.models import utils
from lightly.models.modules.heads import MSNProjectionHead
from lightly.models.modules.masked_autoencoder import MAEBackbone
from lightly.transforms.msn_transform import MSNTransform


class MSN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mask_ratio = 0.15
        self.backbone = MAEBackbone(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=384 * 4,
        )
        self.projection_head = MSNProjectionHead(384)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight

        self.criterion = MSNLoss(gather_distributed=True)

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views, _, _ = batch
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(targets)
        targets_out = self.projection_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        return loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(anchors, idx_keep)
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optim = torch.optim.AdamW(params, lr=1.5e-4)
        return optim


model = MSN()

dataset = LightlyDataset.from_torch_dataset('path/to/dataset', transform=MSNTransform())

collate_fn = MultiViewCollate()

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

trainer = pl.Trainer(
    max_epochs=10,
    devices="auto",
    accelerator="gpu",
    strategy="ddp",
    use_distributed_sampler=True,
)
trainer.fit(model=model, train_dataloaders=dataloader)