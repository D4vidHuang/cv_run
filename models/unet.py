import math
import torch
import torch.nn as nn
from torch import fft


# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

# class DiffusionUNet_frequency(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
#         num_res_blocks = config.model.num_res_blocks
#         attn_resolutions = config.model.attn_resolutions
#         dropout = config.model.dropout
#         in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
#         resolution = config.data.image_size
#         resamp_with_conv = config.model.resamp_with_conv
#
#         self.b1 = 1.6
#         self.b2 = 1.5
#         self.s1 = 0.8
#         self.s2 = 0.8
#
#         self.threshold1 = 3
#         self.threshold2 = 5
#
#         self.ch = ch
#         self.temb_ch = self.ch*4
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels
#
#         # timestep embedding
#         self.temb = nn.Module()
#         self.temb.dense = nn.ModuleList([
#             torch.nn.Linear(self.ch,
#                             self.temb_ch),
#             torch.nn.Linear(self.temb_ch,
#                             self.temb_ch),
#         ])
#
#         # downsampling
#         self.conv_in = torch.nn.Conv2d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)
#
#         curr_res = resolution
#         in_ch_mult = (1,)+ch_mult
#         self.down = nn.ModuleList()
#         block_in = None
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)
#
#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#
#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             skip_in = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 if i_block == self.num_res_blocks:
#                     skip_in = ch*in_ch_mult[i_level]
#                 block.append(ResnetBlock(in_channels=block_in+skip_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up)  # prepend to get consistent order
#
#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in,
#                                         out_ch,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)
#
#     def forward(self, x, t):
#         assert x.shape[2] == x.shape[3] == self.resolution
#
#         print(x.shape)
#         print(self.resolution)
#         # timestep embedding
#         temb = get_timestep_embedding(t, self.ch)
#         temb = self.temb.dense[0](temb)
#         temb = nonlinearity(temb)
#         temb = self.temb.dense[1](temb)
#
#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))
#
#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)
#
#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 if(i_level == range(self.num_resolutions)[0]):
#                     skip_h = hs.pop()
#                     print('h1')
#                     # print('h', h.shape)
#                     # print('skip', skip_h.shape)
#
#                     # x_avg = h.mean(dim=0, keepdim=True)  # Shape: (1, H, W)
#                     # min_val = x_avg.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Scalar min per map
#                     # max_val = x_avg.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Scalar max per map
#                     # hidden_mean = (x_avg - min_val) / (max_val - min_val + 1e-8)
#                     # h[:h.shape[0] // 2] = h[:h.shape[0] // 2] * ((self.b1 - 1) * hidden_mean + 1)
#                     hidden_mean = h.mean(1).unsqueeze(1)
#                     B = hidden_mean.shape[0]
#                     hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#                     hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#                     hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
#                                 hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
#
#                     h[:, :h.shape[1]//2] = h[:, :h.shape[1]//2] * ((self.b1 - 1) * hidden_mean + 1)
#                     skip_h = self.Fourier_filter_free(skip_h, threshold=self.threshold1, scale=self.s1)
#                     h = self.up[i_level].block[i_block](
#                         torch.cat([h, skip_h], dim=1), temb)
#                 elif (i_level == range(self.num_resolutions)[1]):
#                     skip_h = hs.pop()
#                     print('h2')
#                     # print('h2', h.shape)
#                     # print('skip2', skip_h.shape)
#
#                     # x_avg = h.mean(dim=0, keepdim=True)  # Shape: (1, H, W)
#                     # min_val = x_avg.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Scalar min per map
#                     # max_val = x_avg.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Scalar max per map
#                     # hidden_mean = (x_avg - min_val) / (max_val - min_val + 1e-8)
#                     # h[:h.shape[0] // 2] = h[:h.shape[0] // 2] * ((self.b2 - 1) * hidden_mean + 1)
#                     hidden_mean = h.mean(1).unsqueeze(1)
#                     B = hidden_mean.shape[0]
#                     hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#                     hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#                     hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
#                             hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
#
#                     h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((self.b2 - 1) * hidden_mean + 1)
#                     skip_h = self.Fourier_filter_free(skip_h, threshold=self.threshold2, scale=self.s2)
#                     h = self.up[i_level].block[i_block](
#                         torch.cat([h, skip_h], dim=1), temb)
#                 else:
#                     h = self.up[i_level].block[i_block](
#                         torch.cat([h, hs.pop()], dim=1), temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)
#
#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h
#
#     def Fourier_filter_free(self, x, threshold, scale):
#         # FFT
#         x_freq = fft.fftn(x, dim=(-2, -1))
#         x_freq = fft.fftshift(x_freq, dim=(-2, -1))
#
#         B, C, H, W = x_freq.shape
#         print('shape',x_freq.shape)
#
#         #version1
#         mask = torch.ones((B, C, H, W)).cuda()
#         crow, ccol = H // 2, W //2
#         mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
#
#
#         # #version 2
#         # mask = torch.ones((B, C, H, W)).cuda() * scale
#         #
#         # # Set the low-frequency region (center) to 1 (no scaling)
#         # crow, ccol = H // 2, W // 2
#         # threCrow = int(threshold * crow)
#         # threCcol = int(threshold * ccol)
#         # mask[..., crow - threCrow:crow + threCrow, ccol - threCcol:ccol + threCcol] = 1.0
#
#         x_freq = x_freq * mask
#         # IFFT
#         x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
#         x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
#
#         return x_filtered
#
#     def Fourier_filter_free_3d(self, x, threshold, scale):
#         """
#         Apply a Fourier-based low-pass filter to a 3D tensor (C, H, W).
#
#         Args:
#             x (torch.Tensor): Input tensor of shape (C, H, W).
#             threshold (int): The cutoff threshold for the low-pass filter.
#             scale (float): Scaling factor for the filtered frequencies.
#
#         Returns:
#             torch.Tensor: Filtered tensor of shape (C, H, W).
#         """
#         # FFT
#         x_freq = fft.fftn(x, dim=(-2, -1))  # Compute FFT along spatial dimensions
#         x_freq = fft.fftshift(x_freq, dim=(-2, -1))  # Shift zero frequency to the center
#
#         C, H, W = x_freq.shape
#
#         #version 2
#         # mask = torch.ones((C, H, W), device=x.device) * scale
#         #
#         # # Set the low-frequency region (center) to 1 (no scaling)
#         # crow, ccol = H // 2, W // 2
#         # threCrow = int(threshold * crow)
#         # threCcol = int(threshold * ccol)
#         # mask[..., crow - threCrow:crow + threCrow, ccol - threCcol:ccol + threCcol] = 1.0
#
#         #Define central region
#         mask = torch.ones((C, H, W), device=x.device)  # Create mask of shape (C, H, W)
#         crow, ccol = H // 2, W // 2
#         mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
#
#
#         x_freq = x_freq * mask.clone()  # Apply frequency mask
#
#         # IFFT
#         x_freq = fft.ifftshift(x_freq, dim=(-2, -1))  # Shift back
#         x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real  # Compute inverse FFT
#
#         return x_filtered

class DiffusionUNet_frequency(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        # Trainable parameters
        self.b1 = nn.Parameter(torch.tensor(1.6, dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor(1.5, dtype=torch.float32))
        self.s1 = nn.Parameter(torch.randn(4, 128, 64, 64, dtype=torch.float32))  # Learnable tensor
        self.s2 = nn.Parameter(torch.randn(4, 128, 32, 32, dtype=torch.float32))  # Learnable tensor
        # self.threshold1 = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        # self.threshold2 = nn.Parameter(torch.tensor(5.0, dtype=torch.float32))
        self.threshold1 = 1.0
        self.threshold2 = 1.0

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        print(x.shape)
        print(self.resolution)
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                if (i_level == range(self.num_resolutions)[0]):
                    skip_h = hs.pop()
                    # print('h1')
                    # print('h', h.shape)
                    # print('skip', skip_h.shape)

                    # x_avg = h.mean(dim=0, keepdim=True)  # Shape: (1, H, W)
                    # min_val = x_avg.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Scalar min per map
                    # max_val = x_avg.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Scalar max per map
                    # hidden_mean = (x_avg - min_val) / (max_val - min_val + 1e-8)
                    # h[:h.shape[0] // 2] = h[:h.shape[0] // 2] * ((self.b1 - 1) * hidden_mean + 1)
                    hidden_mean = h.mean(1).unsqueeze(1)
                    B = hidden_mean.shape[0]
                    hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                    hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                    hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
                            hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                    h = h.clone()
                    h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2].clone() * ((self.b1 - 1) * hidden_mean + 1)
                    skip_h = self.Fourier_filter_free(skip_h, threshold=self.threshold1, scale=self.s1)
                    h = self.up[i_level].block[i_block](
                        torch.cat([h, skip_h], dim=1), temb)
                elif (i_level == range(self.num_resolutions)[1]):
                    skip_h = hs.pop()
                    # print('h2')
                    # print('h2', h.shape)
                    # print('skip2', skip_h.shape)

                    # x_avg = h.mean(dim=0, keepdim=True)  # Shape: (1, H, W)
                    # min_val = x_avg.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]  # Scalar min per map
                    # max_val = x_avg.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # Scalar max per map
                    # hidden_mean = (x_avg - min_val) / (max_val - min_val + 1e-8)
                    # h[:h.shape[0] // 2] = h[:h.shape[0] // 2] * ((self.b2 - 1) * hidden_mean + 1)
                    hidden_mean = h.mean(1).unsqueeze(1)
                    B = hidden_mean.shape[0]
                    hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                    hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                    hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
                            hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                    h = h.clone()
                    h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2].clone() * ((self.b2 - 1) * hidden_mean + 1)
                    skip_h = self.Fourier_filter_free(skip_h, threshold=self.threshold2, scale=self.s2)
                    h = self.up[i_level].block[i_block](
                        torch.cat([h, skip_h], dim=1), temb)
                else:
                    h = self.up[i_level].block[i_block](
                        torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def Fourier_filter_free(self, x, threshold, scale):
        """Apply a learnable Fourier-based low-pass filter."""
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        # B, C, H, W = x_freq.shape
        # mask = torch.ones((B, C, H, W), device=x.device)
        # crow, ccol = H // 2, W // 2
        # mask[..., crow - int(threshold.item()):crow + int(threshold.item()),
        #      ccol - int(threshold.item()):ccol + int(threshold.item())] = scale
        #
        # x_freq = x_freq * mask
        # print(x_freq.shape)
        # print(scale.shape)
        x_freq = x_freq * scale
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

        return x_filtered

