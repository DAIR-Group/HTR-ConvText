import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath
from timm.layers import LayerScale
import numpy as np
from model import resnet18
from functools import partial
import random
import re
import warnings


class RelativePositionBias1D(nn.Module):
    def __init__(self, num_heads: int, max_rel_positions: int = 1024):
        super().__init__()
        self.num_heads = num_heads
        self.max_rel_positions = max(1, int(max_rel_positions))
        self.bias = nn.Embedding(2 * self.max_rel_positions - 1, num_heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, N: int) -> torch.Tensor:
        device = self.bias.weight.device
        coords = torch.arange(N, device=device)
        rel = coords[:, None] - coords[None, :]
        rel = rel.clamp(-self.max_rel_positions + 1,
                        self.max_rel_positions - 1)
        rel = rel + (self.max_rel_positions - 1)
        bias = self.bias(rel)
        return bias.permute(2, 0, 1).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        max_rel_positions = max(
            1, int(num_patches)) if num_patches is not None else 1024
        self.rel_pos_bias = RelativePositionBias1D(
            num_heads=num_heads, max_rel_positions=max_rel_positions)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos_bias(N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, activation=nn.SiLU):
        super().__init__()
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.act = activation()
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lin2(self.act(self.lin1(x))))


class ConvModule(nn.Module):
    def __init__(self, dim, kernel_size=3, dropout=0.1, drop_path=0.0,
                 expansion=1.0, pre_norm=False, activation=nn.SiLU):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim) if pre_norm else None
        hidden = int(round(dim * expansion))

        self.pw1 = nn.Conv1d(dim, hidden, kernel_size=1, bias=True)
        self.act1 = activation()

        self.dw = nn.Conv1d(hidden, hidden, kernel_size=kernel_size,
                            padding=kernel_size // 2, groups=hidden, bias=True)
        self.gn = nn.GroupNorm(1, hidden, eps=1e-5)
        self.act2 = activation()

        self.pw2 = nn.Conv1d(hidden, dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        z = x.transpose(1, 2)
        z = self.pw1(z)
        z = self.act1(z)
        z = self.dw(z)
        z = self.gn(z)
        z = self.act2(z)
        z = self.pw2(z)
        z = self.dropout(z).transpose(1, 2)
        return self.drop_path(z)


class Downsample1D(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=2, lowpass_init=True):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                            stride=stride, padding=kernel_size//2,
                            groups=dim, bias=False)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        if lowpass_init:
            with torch.no_grad():
                w = torch.zeros_like(self.dw.weight)
                w[:, 0, :] = 1.0 / kernel_size
                self.dw.weight.copy_(w)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pw(self.dw(x))
        return x.transpose(1, 2)


class Upsample1D(nn.Module):
    def __init__(self, dim, mode: str = 'nearest'):
        super().__init__()
        assert mode in (
            'nearest', 'linear'), "Upsample1D mode must be 'nearest' or 'linear'"
        self.mode = mode
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x, target_len: int):
        x = x.transpose(1, 2)
        if self.mode == 'nearest':
            x = F.interpolate(x, size=target_len, mode='nearest')
        else:
            x = F.interpolate(x, size=target_len,
                              mode='linear', align_corners=False)
        x = self.proj(x)
        return x.transpose(1, 2)


class ConvTextBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 num_patches,
                 mlp_ratio=4.0,
                 ff_dropout=0.1,
                 attn_dropout=0.0,
                 conv_dropout=0.0,
                 conv_kernel_size=3,
                 conv_expansion=1.0,
                 norm_layer=nn.LayerNorm,
                 drop_path=0.0,
                 layerscale_init=1e-5):
        super().__init__()

        ff_hidden = int(dim * mlp_ratio)

        self.attn = Attention(dim, num_patches, num_heads=num_heads,
                              qkv_bias=True, attn_drop=attn_dropout, proj_drop=ff_dropout)

        self.ffn1 = FeedForward(
            dim, ff_hidden, dropout=ff_dropout, activation=nn.SiLU)
        self.conv = ConvModule(dim, kernel_size=conv_kernel_size,
                               dropout=conv_dropout, drop_path=0.0,
                               expansion=conv_expansion, pre_norm=False, activation=nn.SiLU)
        self.ffn2 = FeedForward(
            dim, ff_hidden, dropout=ff_dropout, activation=nn.SiLU)

        self.postln_attn = norm_layer(dim, elementwise_affine=True)
        self.postln_ffn1 = norm_layer(dim, elementwise_affine=True)
        self.postln_conv = norm_layer(dim, elementwise_affine=True)
        self.postln_ffn2 = norm_layer(dim, elementwise_affine=True)

        self.dp_attn = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp_ffn1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp_conv = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.dp_ffn2 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls_attn = LayerScale(dim, init_values=layerscale_init)
        self.ls_ffn1 = LayerScale(dim, init_values=layerscale_init)
        self.ls_conv = LayerScale(dim, init_values=layerscale_init)
        self.ls_ffn2 = LayerScale(dim, init_values=layerscale_init)

    def forward(self, x):
        x = self.postln_attn(x + self.ls_attn(self.dp_attn(self.attn(x))))
        x = self.postln_ffn1(
            x + self.ls_ffn1(0.5 * self.dp_ffn1(self.ffn1(x))))
        x = self.postln_conv(x + self.ls_conv(self.dp_conv(self.conv(x))))
        x = self.postln_ffn2(
            x + self.ls_ffn2(0.5 * self.dp_ffn2(self.ffn2(x))))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class HTR_ConvText(nn.Module):
    def __init__(
        self,
        nb_cls=80,
        img_size=[512, 64],
        patch_size=[4, 32],
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        down_after: int = 2,
        up_after: int = 4,
        ds_kernel: int = 3,
        max_seq_len: int = 1024,
        upsample_mode: str = 'nearest',
    ):
        super().__init__()

        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.embed_dim = embed_dim
        self.max_rel_pos = int(max_seq_len)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            ConvTextBlock(embed_dim, num_heads, self.max_rel_pos,
                               mlp_ratio=mlp_ratio,
                               ff_dropout=dropout, attn_dropout=dropout,
                               conv_dropout=dropout, conv_kernel_size=conv_kernel_size,
                               conv_expansion=1.0,
                               norm_layer=norm_layer, drop_path=dpr[i],
                               layerscale_init=1e-5)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        self.head = torch.nn.Linear(embed_dim, nb_cls)
        self.down_after = down_after
        self.up_after = up_after
        self.down1 = Downsample1D(embed_dim, kernel_size=ds_kernel)
        self.up1 = Upsample1D(embed_dim, mode=upsample_mode)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_random_1d(self, x, ratio):
        B, L, _ = x.shape
        mask = torch.ones(B, L, dtype=torch.bool).to(x.device)
        if ratio <= 0.0 or ratio > 1.0:
            return mask
        num = int(round(ratio * L))
        if num <= 0:
            return mask
        noise = torch.rand(B, L).to(x.device)
        idx = noise.argsort(dim=1)[:, :num]
        mask.scatter_(1, idx, False)
        return mask

    def mask_block_1d(self, x, ratio: float, max_block_length: int):
        B, L, _ = x.shape
        device = x.device

        if ratio <= 0.0:
            return torch.ones(B, L, 1, dtype=torch.bool, device=device)
        if ratio >= 1.0:
            return torch.zeros(B, L, 1, dtype=torch.bool, device=device)

        target_mask_tokens = int(round(ratio * L))
        K = target_mask_tokens // max_block_length
        K = max(K, 1)
        starts = torch.randint(0, max(1, L - max_block_length + 1), (B, K), device=device)
        lengths = torch.randint(1, max_block_length + 1, (B, K), device=device)
        positions = torch.arange(L, device=device).view(1, 1, L)
        starts_exp = starts.unsqueeze(-1)
        ends_exp = (starts + lengths).unsqueeze(-1).clamp(max=L)
        blocks_mask = (positions >= starts_exp) & (positions < ends_exp)
        masked_any = blocks_mask.any(dim=1)
        keep_mask = ~masked_any
        return keep_mask.unsqueeze(-1)


    def mask_span_1d(self, x, ratio: float, max_span_length: int):
        B, L, _ = x.shape
        device = x.device

        if ratio <= 0.0:
            return torch.ones(B, L, 1, dtype=torch.bool, device=device)
        if ratio >= 1.0:
            return torch.zeros(B, L, 1, dtype=torch.bool, device=device)

        target_mask_tokens = int(round(ratio * L))
        K = target_mask_tokens // max_span_length
        K = max(K, 1)
        starts = torch.randint(0, max(1, L - max_span_length + 1), (B, K), device=device)
        lengths = torch.full((B, K), max_span_length, device=device)
        positions = torch.arange(L, device=device).view(1, 1, L)
        starts_exp = starts.unsqueeze(-1)
        ends_exp = (starts + lengths).unsqueeze(-1).clamp(max=L)
        spans_mask = (positions >= starts_exp) & (positions < ends_exp)
        masked_any = spans_mask.any(dim=1)
        keep_mask = ~masked_any
        return keep_mask.unsqueeze(-1)


    def forward_features(self, x, use_masking=False,
                         mask_mode="span",
                         mask_ratio=0.5, block_span=4, max_span_length=8):
        x = self.patch_embed(x)
        B, C, W, H = x.shape
        assert C == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {C}"
        x = x.view(B, C, -1).permute(0, 2, 1)

        masked_positions_1d = None
        if use_masking:
            if mask_mode == "random":
                keep_mask_1d = self.mask_random_1d(x, mask_ratio).float()
                mask = keep_mask_1d.unsqueeze(-1)
            elif mask_mode in ("block"):
                keep_mask = self.mask_block_1d(x, mask_ratio, block_span).float()
                keep_mask_1d = keep_mask.squeeze(-1)
                mask = keep_mask
            elif mask_mode in ("span"):
                keep_mask = self.mask_span_1d(
                    x, mask_ratio, max_span_length).float()
                keep_mask_1d = keep_mask.squeeze(-1)
                mask = keep_mask
            else:
                warnings.warn(
                    f"Unknown mask_mode '{mask_mode}', defaulting to span.")
                keep_mask = self.mask_span_1d(
                    x, mask_ratio, max_span_length).float()
                keep_mask_1d = keep_mask.squeeze(-1)
                mask = keep_mask
            masked_positions_1d = (1.0 - keep_mask_1d).clamp(min=0.0, max=1.0)
            x = mask * x + (1.0 - mask) * \
                self.mask_token.expand(x.size(0), x.size(1), x.size(2))
        skip_hi = None
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x)
            if i == self.down_after:
                skip_hi = x
                if (x.size(1) % 2) == 1:
                    x = torch.cat([x, x[:, -1:, :]], dim=1)
                x = self.down1(x)
            if i == self.up_after:
                assert skip_hi is not None, "Upsample requires a stored skip."
                x = self.up1(x, target_len=skip_hi.size(1))
                x = x + skip_hi

        x = self.norm(x)
        return x, masked_positions_1d

    def forward(self, x, use_masking=False, return_features=False, return_mask=False,
                mask_mode="span", mask_ratio=None, block_span=None, max_span_length=None):
        feats, masked_positions_1d = self.forward_features(
            x, use_masking=use_masking, mask_mode=mask_mode, mask_ratio=mask_ratio, block_span=block_span, max_span_length=max_span_length)
        logits = self.head(feats)
        if return_features and return_mask:
            return logits, feats, (masked_positions_1d if masked_positions_1d is not None else None)
        if return_features:
            return logits, feats
        if return_mask:
            return logits, (masked_positions_1d if masked_positions_1d is not None else None)
        return logits


def create_model(nb_cls, img_size, mlp_ratio=4, **kwargs):
    model = HTR_ConvText(
        nb_cls,
        img_size=img_size,
        patch_size=(4, 64),
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        conv_kernel_size=7,
        down_after=3,
        up_after=7,
        ds_kernel=3,
        max_seq_len=128,
        upsample_mode='nearest',
        **kwargs,
    )
    return model
