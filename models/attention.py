import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, dim_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim_embed, 3 * dim_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(dim_embed, dim_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = dim_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: BATCH_SIZE, SEQ_LEN, DIM
        input_shape = x.shape
        batch_size, sequence_length, dim_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (BATCH_SIZE, SEQ_LEN, DIM) -> (BATCH_SIZE, SEQ_LEN, DIM * 3) -> 3 tensors of shape (BATCH_SIZE, SEQ_LEN, DIM)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (BATCH_SIZE, SEQ_LEN, DIM) -> (BATCH_SIZE, SEQ_LEN,H, DIM / H) -> (BATCH_SIZE,H, SEQ_LEN, DIM / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=1)

        # (BATCH_SIZE, H, SEQ_LEN, SEQ_LEN) * (BATCH_SIZE, H, SEQ_LEN, DIM / H) -> (BATCH_SIZE, H, SEQ_LEN, DIM / H)
        output = weight @ v

        # (BATCH_SIZE, H, SEQ_LEN, DIM / H) -> (BATCH_SIZE, SEQ_LEN, H, DIM / H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        dim_embed: int,
        dim_cross: int,
        in_proj_bias=True,
        out_proj_bias=True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(dim_embed, dim_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(dim_cross, dim_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(dim_cross, dim_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(dim_embed, dim_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = dim_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x(latent): (BATCH_SIZE, SEQ_LEN_Q, DIM_Q)
        # y(context): (BATCH_SIZE, SEQ_LEN_KV, DIM_KV) = (BATCH_SIZE, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, dim_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output
