import torch
import torch.nn as nn
from lightly.models import utils

# vision_transformer requires torchvision >= 0.12
from torchvision.models import vision_transformer

class MAEDecoder(vision_transformer.Encoder):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        embed_input_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )
        self.decoder_embed = nn.Linear(embed_input_dim, hidden_dim, bias=True)
        self.prediction_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.embed(input)
        out = self.decode(out)
        return self.predict(out)

    def embed(self, input: torch.Tensor) -> torch.Tensor:
        return self.decoder_embed(input)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(input)