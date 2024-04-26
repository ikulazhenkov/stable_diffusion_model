import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttnBlock, VAE_ResidualBlock


class VAE_encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (BATCH_SIZE,CHANNEL,HEIGHT,WIDTH) -> (BATCH_SIZE, 128,HEIGHT, WIDTH)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # input (BATCH_SIZE, 128,HEIGHT, WIDTH) -> (BATCH_SIZE, 128,HEIGHT, WIDTH)
            VAE_ResidualBlock(128, 128),
            # input (BATCH_SIZE, 128,HEIGHT, WIDTH) -> (BATCH_SIZE, 128,HEIGHT, WIDTH)
            VAE_ResidualBlock(128, 128),
            # input (BATCH_SIZE, 128,HEIGHT, WIDTH) -> (BATCH_SIZE, 128,HEIGHT/2, WIDTH/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # input (BATCH_SIZE, 128,HEIGHT/2, WIDTH/2) -> (BATCH_SIZE, 256,HEIGHT/2, WIDTH/2)
            VAE_ResidualBlock(128, 256),
            # input (BATCH_SIZE, 256,HEIGHT/2, WIDTH/2) -> (BATCH_SIZE, 256,HEIGHT/2, WIDTH/2)
            VAE_ResidualBlock(256, 256),
            # input (BATCH_SIZE, 256,HEIGHT/2, WIDTH/2) -> (BATCH_SIZE, 256 ,HEIGHT/4, WIDTH/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # input (BATCH_SIZE, 256,HEIGHT/4, WIDTH/4) -> (BATCH_SIZE, 512,HEIGHT/4, WIDTH/4)
            VAE_ResidualBlock(256, 512),
            # input (BATCH_SIZE, 512,HEIGHT/4, WIDTH/4) -> (BATCH_SIZE, 512,HEIGHT/4, WIDTH/4)
            VAE_ResidualBlock(512, 512),
            # input (BATCH_SIZE, 512,HEIGHT/4, WIDTH/4) -> (BATCH_SIZE, 512 ,HEIGHT/8, WIDTH/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # input (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8) -> (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8)
            VAE_ResidualBlock(512, 512),
            # input (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8) -> (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8)
            VAE_ResidualBlock(512, 512),
            # input (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8) -> (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8)
            VAE_ResidualBlock(512, 512),
            # input (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8) -> (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8)
            VAE_AttnBlock(512),
            VAE_ResidualBlock(512, 512),
            # normalization
            # input (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8) -> (BATCH_SIZE, 512,HEIGHT/8, WIDTH/8)
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (BATCH_SIZE,CHANNEL,HEIGHT,WIDTH)
        # noise: (BATCH_SIZE,OUT_CHANNEL,HEIGHT/8,WIDTH/8)

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(0, 1, 0, 1)
            x = module(x)

        mu, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()

        # (BATCH_SIZE,4,HEIGHT/8,WIDTH/8) -> (BATCH_SIZE,4,HEIGHT/8,WIDTH/8)
        std = variance.sqrt()

        # z=N(0,1) -> N(mean,variance)=X?
        # X=mean + std * Z
        x = mu + std * noise

        # next we scale the output by a constant
        x *= 0.18215

        return x
