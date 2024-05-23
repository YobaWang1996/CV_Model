import torch
from torch import nn
from einops import repeat
from timm.models.layers import trunc_normal_
from UCSDPed2 import UCSDPed2
from attention import PreNorm, Attention, FeedForward
from swin_transformer import SwinTransformer
from torch.utils.data import DataLoader


def build_model():
    model1 = SwinTransformer(img_size=224,
                             patch_size=4,
                             in_chans=3,
                             num_classes=768,
                             embed_dim=96,
                             depths=(2, 2, 6, 2),
                             num_heads=(3, 6, 12, 24),
                             window_size=7,
                             mlp_ratio=4.,
                             qkv_bias=True,
                             drop_rate=0.,
                             attn_drop_rate=0.,
                             drop_path_rate=0.1,
                             ape=False,
                             patch_norm=True,
                             use_checkpoint=False
                             )

    model_dict = model1.state_dict()
    path = ""  # path of pretrained swin transformer
    pretrained_dict = torch.load(path)
    update_dict = {}
    for _, j in pretrained_dict.items():
        update_dict = {k: v for k, v in j.items() if k in model_dict and k != 'head.weight' and k != 'head.bias'}

    model_dict.update(update_dict)
    model1.load_state_dict(model_dict)

    return model1


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class spatial_temporal_Fusion_Module(nn.Module):
    def __init__(self,
                 swinT=build_model(),
                 out_dim=192,
                 dim=768,
                 num_frames=16,
                 depth=4,
                 heads=16,
                 dim_head=48,
                 dropout=0.,
                 ff_dim=4,
                 ape=True):
        super().__init__()

        self.num_frame = num_frames
        self.dim = dim
        self.outdim = out_dim
        self.dim_head = dim_head
        self.spatial_transformer = swinT
        self.ape = ape

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.randn(1, num_frames, dim), requires_grad=True)
        self.pos_drop = nn.Dropout(dropout)

        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * ff_dim, dropout)

        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool2d((1, dim))

        self.head = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.spatial_transformer(x)
        x = torch.reshape(x, (-1, self.num_frame, self.dim))

        if self.ape:
            x += self.absolute_pos_embed
            x = self.pos_drop(x)

        x = self.temporal_transformer(x)
        x = self.norm(x)
        x = self.pool(x).view(-1, self.dim)

        x = self.head(x)
        return x
