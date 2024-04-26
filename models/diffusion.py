import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x - (1,320)

        x = self.linear1(x)

        x = F.silu(x)

        x = self.linear2(x)

        # x - (1,280)
        return x


class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time: 1280):
        super().__init__()
        self.groupnorm_features = nn.GroupNorm(32, in_channels)
        self.conv_features = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_merged = nn.GroupNorm(32, in_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
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

    def forward(self, feature, time):
        # feature: (BATCH_SIZE,IN_CHANNEL,HEIGHT,WIDTH)
        # time: (1,1280)
        residue = feature
        feature = self.groupnorm_features(feature)
        feature = F.silu(feature)
        feature = self.conv_features(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        return merged + self.residual_layer(residue)


class UNet_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context=768):
        super().__init__()
        channels = n_head * n_embed
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm2 = nn.LayerNorm(channels)

        self.attention2 = CrossAttention(
            n_head, channels, d_context, in_proj_bias=False
        )
        self.layernorm3 = nn.LayerNorm(channels)

        self.lineargeglu1 = nn.Linear(channels, 4 * channels * 2)
        self.lineargeglu2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: (BATCH_SIZE,FEATURES,HEIGHT,WIDTH)
        # context: (BATCH_SIZE,SEQ_LEN,DIM)

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        # Normalization + self attention with skip connect
        residue_short = x

        x = self.layernorm1(x)
        self.attention1(x)
        x += residue_short

        residue_short = x
        # Normalization + cross attention with skip connect
        x = self.layernorm2(x)
        self.attention2(x, context)
        x += residue_short

        residue_short = x

        # Normalization + FF with GeGLU and skip connection
        x = self.layernorm3(x)

        x, gate = self.lineargeglu1(x).chunk(2, dim=1)
        x = x * F.gelu(gate)

        x = self.lineargeglu2(x)

        x += residue_short

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        x = self.conv_output(x) + residue_long
        return x


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (BATCH_SIZE, FEATURES, HEIGHT, WIDTH) -> (BATCH_SIZE, FEATURES, HEIGHT * 2, WIDTH * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.Module(
            [
                # (BATCH_SIZE, 4, HEIGHT / 8, WIDTH / 8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                SwitchSequential(
                    UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(320, 320), UNet_AttentionBlock(8, 40)
                ),
                # (BATCH_SIZE, 320, HEIGHT / 8, WIDTH / 8) -> (BATCH_SIZE, 320, HEIGHT / 16, WIDTH / 16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(320, 640), UNet_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(640, 640), UNet_AttentionBlock(8, 80)
                ),
                # (BATCH_SIZE, 640, HEIGHT / 16, WIDTH / 16) -> (BATCH_SIZE, 640, HEIGHT / 32, WIDTH / 32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(640, 1280), UNet_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(1280, 1280), UNet_AttentionBlock(8, 160)
                ),
                # (BATCH_SIZE, 1280, HEIGHT / 32, WIDTH / 32) -> (BATCH_SIZE, 1280, HEIGHT / 64, WIDTH / 64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(UNet_ResidualBlock(1280, 1280)),
                SwitchSequential(UNet_ResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(1280, 1280),
            UNet_AttentionBlock(8, 160),
            UNet_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # (BATCH_SIZE, 2560, HEIGHT / 64, WIDTH / 64) -> (BATCH_SIZE, 1280, HEIGHT / 64, WIDTH / 64)
                SwitchSequential(UNet_ResidualBlock(2560, 1280)),
                SwitchSequential(UNet_ResidualBlock(2560, 1280)),
                SwitchSequential(UNet_ResidualBlock(2560, 1280), UpSample(1280)),
                SwitchSequential(
                    UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(2560, 1280), UNet_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(1920, 1280),
                    UNet_AttentionBlock(8, 160),
                    UpSample(1280),
                ),
                SwitchSequential(
                    UNet_ResidualBlock(1920, 640), UNet_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(1280, 640), UNet_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(960, 640),
                    UNet_AttentionBlock(8, 80),
                    UpSample(640),
                ),
                SwitchSequential(
                    UNet_ResidualBlock(960, 320), UNet_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNet_ResidualBlock(640, 320), UNet_AttentionBlock(8, 40)
                ),
            ]
        )


class UNetOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (BATCH_SIZE,320,HEIGHT/8,WIDTH/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        return x


class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetOutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent - (BATCH_SIZE,4,HEIGHT/8,WIDTH/8)
        # context - (BATCH_SIZE,SEQ_LEN,DIM) DIM=768
        # time    -  (1,320)

        # (1,320) -> (1,1280)
        time = self.time_embedding(time)

        # (BATCH_SIZE,4,HEIGHT/8,WIDTH/8) -> (BATCH_SIZE,320,HEIGHT/8,WIDTH/8)
        output = self.unet(latent, context, time)

        output = self.final(output)

        output = self.final(output)

        return output
