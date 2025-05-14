import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from timm.models.layers import to_2tuple, trunc_normal_
from typing import Tuple
from einops import rearrange, repeat
from torchsummary import summary
import math

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)     (窗口数量*Batch_size,窗口大小，窗口大小，通道数)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)     #划分窗口
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x  


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim   #线性投影后的维度
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape       #解析输入维度

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

#下采样 （宽高减半，通道加倍）
class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        #x = x.view(B, C, H, W)  # reshape to [B, C, H, W]
        x = x.reshape(B, C, H, W) 
        x = self.conv(x)  # apply convolution

        x = x.view(B, -1, 2 * C)  # reshape back to [B, H/2*W/2, 2*C]
       
        return x

#条件位置编码
# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        #卷积层用于提取位置特征，卷积核大小为3x3,步长为s,组数等于输出通道数embed_dim,这样可以看作是1x1卷积
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim) )
        self.s = s

    #对输入维度为B*N*C的token系序列进行reshape操作，将其还原成2D的shape
    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:        #则self.proj输出直接加上原特征图实现残差连接
            x = self.proj(cnn_feat) + cnn_feat    # CNN聚合局部关系并配合残差连接
        else:
            x = self.proj(cnn_feat)       #否则直接使用self.proj输出
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]    


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)     #
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)     #

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B_, N, C = x.shape      # x: input features with shape of (num_windows*B, Mh*Mw, C)
       
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        #q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)    

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        # print('x shape:', x.shape)
        return x

class ShuffleAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shuffle=False, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., relative_pos_embedding=False):
        super().__init__()
        self.num_heads = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.shuffle = shuffle

        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        b, c, h, w = x.shape
        # print(f"Input shape: {x.shape}")

        
        if h <= self.ws and w <= self.ws:
           
            qkv = self.to_qkv(x)
            # print(f"qkv shape: {qkv.shape}")
            try:
                q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
                                    h=self.num_heads, qkv=3, ws1=h, ws2=w)
                # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
            except Exception as e:
                # print(f"Error during rearrange: {e}")
                raise

            dots = (q @ k.transpose(-2, -1)) * self.scale
            attn = dots.softmax(dim=-1)
            out = attn @ v

            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
                            h=self.num_heads, b=b, hh=1, ws1=h, ws2=w)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out

       
        region_h, region_w = h // 2, w // 2
        regions = [
            x[:, :, 0:region_h, 0:region_w],  # Top-left
            x[:, :, 0:region_h, region_w:w],  # Top-right
            x[:, :, region_h:h, 0:region_w],  # Bottom-left
            x[:, :, region_h:h, region_w:w],  # Bottom-right
        ]
        # for i, region in enumerate(regions):
        # # print(f"Region {i} shape: {region.shape}")

        outputs = []
        for region in regions:
            qkv = self.to_qkv(region)
            # print(f"qkv shape: {qkv.shape}")
            try:
                q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
                                    h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
                # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
            except Exception as e:
                # print(f"Error during rearrange: {e}")
                raise

            dots = (q @ k.transpose(-2, -1)) * self.scale
            attn = dots.softmax(dim=-1)
            out = attn @ v


            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
                            h=self.num_heads, b=b, hh=region.shape[2] // self.ws, ws1=self.ws, ws2=self.ws)
            outputs.append(out)

        top = torch.cat([outputs[0], outputs[1]], dim=-1)  # Concatenate top-left and top-right
        bottom = torch.cat([outputs[2], outputs[3]], dim=-1)  # Concatenate bottom-left and bottom-right
        out = torch.cat([top, bottom], dim=-2)  # Concatenate top and bottom

        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class CrossWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SpatialBlock(nn.Module):
    r""" Spatial-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True,drop=0,attn_drop=0, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.num_regions = 4     #4个区域

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size,self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        #WindowShuffle
        self.shuffle_attn = ShuffleAttention(
            dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        
        self.cross_attn = CrossWindowAttention(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        H, W = self.H,self.W       
        B, L, C = x.shape
        #H = W = int(math.sqrt(L))
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(shortcut)
        #x = x.view(B, H * 2, W * 2, C)
        x = x.view(B, H, W, C)

        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape    #Hp, Wp代表padding后的H，W
        
        
        x_windows = window_partition(x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)   # [nW*B, Mh*Mw, C]
        
        attn_windows = self.attn(x_windows)    #窗口注意力
        # print(f"attn_windows shape: {attn_windows.shape}")

        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        
        # x = window_reverse(attn_windows, self.window_size, H, W)   #Hp,Hw表示padding后窗口的宽高
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)


        if pad_r > 0 or pad_b > 0:
             # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()     #B, H, W, C
        # print(f"x shape: {x.shape}")

         # 使用 ShuffleAttention
        x = self.shuffle_attn(x.permute(0, 3, 1, 2))  # 转换为 [B, C, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # 转换回 [B, H, W, C]


        if H <= self.window_size and W <= self.window_size:

            x = x.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:

            region_h, region_w = H // 2, W // 2
            regions = [
                x[:, 0:region_h, 0:region_w, :],  # Top-left
                x[:, 0:region_h, region_w:W, :],  # Top-right
                x[:, region_h:H, 0:region_w, :],  # Bottom-left
                x[:, region_h:H, region_w:W, :]   # Bottom-right
            ]

            outputs = []
            for region in regions:
                # 划分窗口
                windows = window_partition(region, self.window_size)  # [nW*B, Mh, Mw, C]
                windows = windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

                selected_window = windows[0:1]  # 选择第一个窗口

                # 执行 CrossWindowAttention
                cross_attn_output = self.cross_attn(selected_window)  # [1, Mh*Mw, C]

                new_windows = windows.clone()
                new_windows[0:1] = cross_attn_output

                region_output = window_reverse(windows.view(-1, self.window_size, self.window_size, C),
                                            self.window_size, region.shape[1], region.shape[2])
                outputs.append(region_output)

            # 合并区域
            top = torch.cat([outputs[0], outputs[1]], dim=2)  # Concatenate top-left and top-right
            bottom = torch.cat([outputs[2], outputs[3]], dim=2)  # Concatenate bottom-left and bottom-right
            x = torch.cat([top, bottom], dim=1)  # Concatenate top and bottom

        
        x = x.view(B, H * W, C)

        #FFNswin
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   


class ChannelAttention(nn.Module):
    r""" Channel based self attention.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of the groups.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True,attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads        #组数
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)     #
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)     #

    def forward(self, x):
        B, N, C = x.shape
        group_dim = C // self.num_heads   #分组

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        '''
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        '''
        x = self.proj(x)
        x = self.proj_drop(x)

        # Add a fusion operation here
        #x = x.view(B, N, self.num_heads, group_dim).mean(dim=2).view(B, N, C)

        return x

class ChannelBlock(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,drop=0,attn_drop=0,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x):        
        shortcut = x
        x= self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        '''
        shortcut = x
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = shortcut     
        '''
        x = x + shortcut    #大残差
        return x

class Stage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
                
        # build blocks 
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i % 2 == 0:   #索引i的奇偶性来决定添加SpatialBlock还是ChannelBlock
                self.blocks.append(
                    SpatialBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer)
                )
            else:
                self.blocks.append(
                    ChannelBlock(
                        dim=dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer)
                )
            # 在每个stage的第一个Block之后插入条件位置编码（PosCNN）
            if i == 0:
                self.blocks.append(PosCNN(in_chans=dim, embed_dim=dim, s=1))
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self,x,H,W):         

        for i, blk in enumerate(self.blocks):         #for循环交替使用两种注意力
            if isinstance(blk, PosCNN):
                x = blk(x, H, W)
            else:
                blk.H, blk.W = H, W
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x, H, W)   
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


