import torch
import torch.nn as nn
from lightly.models import utils

# vision_transformer requires torchvision >= 0.12
from torchvision.models import vision_transformer


class MAEEncoder(vision_transformer.Encoder):
    @classmethod
    def from_vit_encoder(cls, vit_encoder: vision_transformer.Encoder):
        encoder = cls(
            seq_length=1,
            num_layers=1,
            num_heads=1,
            hidden_dim=1,
            mlp_dim=1,
            dropout=0,
            attention_dropout=0,
        )
        encoder.pos_embedding = vit_encoder.pos_embedding
        encoder.dropout = vit_encoder.dropout
        encoder.layers = vit_encoder.layers
        encoder.ln = vit_encoder.ln
        return encoder

    def forward(
        self, 
        input: torch.Tensor, 
        idx_keep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        input = input + self.pos_embedding
        if idx_keep is not None:
            input = utils.get_at_index(input, idx_keep)
        return self.ln(self.layers(self.dropout(input)))