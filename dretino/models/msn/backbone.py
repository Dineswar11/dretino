import torch
import torch.nn as nn
from lightly.models import utils

# vision_transformer requires torchvision >= 0.12
from torchvision.models import vision_transformer

class MAEBackbone(vision_transformer.VisionTransformer):
    @classmethod
    def from_vit(cls, vit: vision_transformer.VisionTransformer) -> MAEBackbone:
        backbone = cls(
            image_size=vit.image_size,
            patch_size=vit.patch_size,
            num_layers=1,
            num_heads=1,
            hidden_dim=vit.hidden_dim,
            mlp_dim=vit.mlp_dim,
            dropout=vit.dropout,
            attention_dropout=vit.attention_dropout,
            num_classes=vit.num_classes,
            representation_size=vit.representation_size,
            norm_layer=vit.norm_layer,
        )
        backbone.conv_proj = vit.conv_proj
        backbone.class_token = vit.class_token
        backbone.seq_length = vit.seq_length
        backbone.heads = vit.heads
        backbone.encoder = MAEEncoder.from_vit_encoder(vit.encoder)
        return backbone

    def forward(
        self, 
        images: torch.Tensor, 
        idx_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.encode(images, idx_keep)
        class_token = out[:, 0]
        return class_token

    def encode(
        self, 
        images: torch.Tensor, 
        idx_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.images_to_tokens(images)
        out = utils.prepend_class_token(out, self.class_token)
        return self.encoder(out, idx_keep)

    def images_to_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(images)
        return x.flatten(2).transpose(1, 2) 