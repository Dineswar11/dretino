class ProjectionHead(nn.Module):
    def __init__(
        self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.
        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        return self.layers(x)

class MSNProjectionHead(ProjectionHead):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 2048,
        output_dim: int = 256,
    ):
        super().__init__(
            blocks=[
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.GELU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.GELU()),
                (hidden_dim, output_dim, None, None),
            ]
        )