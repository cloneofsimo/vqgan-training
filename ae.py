# Take from FLUX


import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from utils import wavelet_transform_multi_channel


def swish(x) -> Tensor:
    return x * torch.sigmoid(x)


# class StandardizedC2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.step = 0

#     def forward(self, input):
#         output = super().forward(input)
#         # normalize the weights
#         if self.step < 1000:
#             with torch.no_grad():
#                 std = output.std().item()
#                 normalize_term = (std + 1e-6)**(100/(self.step + 100))
#                 self.step += 1
#                 self.weight.data.div_(normalize_term)
#                 self.bias.data.div_(normalize_term)
#                 output.div_(normalize_term)
#                 # sync the weights, braodcast
#                 torch.distributed.broadcast(self.weight.data, 0)
#                 torch.distributed.broadcast(self.bias.data, 0)

#         return output
StandardizedC2d = nn.Conv2d


class FP32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.head_dim = 64
        self.num_heads = in_channels // self.head_dim
        self.norm = FP32GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.qkv = StandardizedC2d(
            in_channels, in_channels * 3, kernel_size=1, bias=False
        )
        self.proj_out = StandardizedC2d(
            in_channels, in_channels, kernel_size=1, bias=False
        )
        nn.init.normal_(self.proj_out.weight, std=0.2 / math.sqrt(in_channels))

    def attention(self, h_) -> Tensor:
        h_ = self.norm(h_)
        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)
        b, c, h, w = q.shape
        q = rearrange(
            q, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        k = rearrange(
            k, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        v = rearrange(
            v, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return h_

    def forward(self, x) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = FP32GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = StandardizedC2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = FP32GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = StandardizedC2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = StandardizedC2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

        # init conv2 as very small number
        nn.init.normal_(self.conv2.weight, std=0.0001 / self.out_channels)
        nn.init.zeros_(self.conv2.bias)
        self.counter = 0

    def forward(self, x):

        # if self.counter < 5000:
        #     self.counter += 1
        #     h = 0
        # else:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = StandardizedC2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = StandardizedC2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        use_attn: bool = True,
        use_wavelet: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_wavelet = use_wavelet
        if self.use_wavelet:
            self.wavelet_transform = wavelet_transform_multi_channel
            self.conv_in = StandardizedC2d(
                4 * in_channels, self.ch * 2, kernel_size=3, stride=1, padding=1
            )
            ch_mult[0] *= 2
        else:
            self.wavelet_transform = nn.Identity()
            self.conv_in = StandardizedC2d(
                in_channels, self.ch, kernel_size=3, stride=1, padding=1
            )

        curr_res = resolution
        in_ch_mult = (2 if self.use_wavelet else 1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1 and not (
                self.use_wavelet and i_level == 0
            ):
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in) if use_attn else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = FP32GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = StandardizedC2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )
        for module in self.modules():
            if isinstance(module, StandardizedC2d):
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.GroupNorm):
                nn.init.zeros_(module.bias)

    def forward(self, x) -> Tensor:
        h = self.wavelet_transform(x)
        h = self.conv_in(h)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1 and not (
                self.use_wavelet and i_level == 0
            ):
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        use_attn: bool = True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        self.conv_in = StandardizedC2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in) if use_attn else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = FP32GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = StandardizedC2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

        # initialize all bias to zero
        for module in self.modules():
            if isinstance(module, StandardizedC2d):
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.GroupNorm):
                nn.init.zeros_(module.bias)

    def forward(self, z) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z) -> Tensor:
        mean = z
        if self.sample:
            std = 0.00
            return mean * (1 + std * torch.randn_like(mean))
        else:
            return mean


class VAE(nn.Module):
    def __init__(
        self,
        resolution,
        in_channels,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        z_channels,
        use_attn,
        decoder_also_perform_hr,
        use_wavelet,
    ):
        super().__init__()
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            use_attn=use_attn,
            use_wavelet=use_wavelet,
        )
        self.decoder = Decoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult + [4] if decoder_also_perform_hr else ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            use_attn=use_attn,
        )
        self.reg = DiagonalGaussian()

    def forward(self, x) -> Tensor:
        z = self.encoder(x)
        z_s = self.reg(z)
        decz = self.decoder(z_s)
        return decz, z


if __name__ == "__main__":
    from utils import prepare_filter

    prepare_filter("cuda")
    vae = VAE(
        resolution=256,
        in_channels=3,
        ch=64,
        out_ch=3,
        ch_mult=[1, 2, 4, 4, 4],
        num_res_blocks=2,
        z_channels=16 * 4,
        use_attn=False,
        decoder_also_perform_hr=False,
        use_wavelet=False,
    )
    vae.eval().to("cuda")
    x = torch.randn(1, 3, 256, 256).to("cuda")
    decz, z = vae(x)
    print(decz.shape, z.shape)

    # do de

    # from unit_activation_reinitializer import adjust_weight_init
    # from torchvision import transforms
    # import torchvision

    # train_dataset = torchvision.datasets.CIFAR10(
    #     root="./data",
    #     train=True,
    #     download=True,
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize((256, 256)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]
    #     ),
    # )

    # initial_std, layer_weight_std = adjust_weight_init(
    #     vae,
    #     dataset=train_dataset,
    #     device="cuda:0",
    #     batch_size=64,
    #     num_workers=0,
    #     tol=0.1,
    #     max_iters=10,
    #     exclude_layers=[FP32GroupNorm, nn.LayerNorm],
    # )

    # # save initial_std and layer_weight_std
    # torch.save(initial_std, "initial_std.pth")
    # torch.save(layer_weight_std, "layer_weight_std.pth")

    # print("\nAdjusted Weight Standard Deviations. Before -> After:")
    # for layer_name, std in layer_weight_std.items():
    #     print(
    #         f"Layer {layer_name}, Changed STD from \n   {initial_std[layer_name]:.4f} -> STD {std:.4f}\n"
    #     )

    # print(layer_weight_std)
