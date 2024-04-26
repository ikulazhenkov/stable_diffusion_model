import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH_SIZE,IN_CHANNEL,HEIGHT,WIDTH)
        residue = x
        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=4, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttnBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (BATCH_SIZE, 512, HEIGHT / 8, WIDTH / 8) -> (BATCH_SIZE, 512, HEIGHT / 4, WIDTH / 4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (BATCH_SIZE, 256, HEIGHT / 4, WIDTH / 4) -> (BATCH_SIZE, 256, HEIGHT / 2, WIDTH / 2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (BATCH_SIZE, 128, HEIGHT / 2, WIDTH / 2) -> (BATCH_SIZE, 128, HEIGHT, WIDTH)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # (BATCH_SIZE, 128, HEIGHT, WIDTH) -> (BATCH_SIZE, 3, HEIGHT, WIDTH)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH_SIZE, 4, HEIGHT / 8, WIDTH / 8)

        x /= 0.18215

        for module in self:
            x = module(x)

        return x


class VAE_AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, num_channels=channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH_SIZE,FEATURES,HEIGHT,WIDTH)
        residue = x

        n, c, h, w = x.shape

        # (BATCH_SIZE,FEATURES,HEIGHT,WIDTH) -> (BATCH_SIZE,FEATURES,HEIGHT*WIDTH)
        x = x.view(n, c, h * w)

        # (BATCH_SIZE,FEATURES,HEIGHT*,WIDTH) -> (BATCH_SIZE,HEIGHT*WIDTH, FEATURES)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # (BATCH_SIZE,HEIGHT*WIDTH, FEATURES) -> (BATCH_SIZE,FEATURES,HEIGHT*,WIDTH)
        x = x.transpose(-1, -2)

        x = x.view(n, c, h, w)

        x += residue

        return x
