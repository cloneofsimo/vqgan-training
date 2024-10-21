from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        try:
            data = torch.load("vgg.pth", map_location=torch.device("cpu"))
        except:
            print("Failed to load vgg.pth, downloading...")
            os.system(
                "wget https://heibox.uni-heidelberg.de/seafhttp/files/9535cbee-6558-4c0c-8743-78f5e56ea75e/vgg.pth"
            )
            data = torch.load("vgg.pth", map_location=torch.device("cpu"))

        self.load_state_dict(
            data,
            strict=False,
        )

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.scaling_layer = ScalingLayer()

        _vgg = models.vgg16(pretrained=True)

        self.slice1 = nn.Sequential(_vgg.features[:4])
        self.slice2 = nn.Sequential(_vgg.features[4:9])
        self.slice3 = nn.Sequential(_vgg.features[9:16])
        self.slice4 = nn.Sequential(_vgg.features[16:23])
        self.slice5 = nn.Sequential(_vgg.features[23:30])

        self.binary_classifier1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier1[-1].weight)

        self.binary_classifier2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier2[-1].weight)

        self.binary_classifier3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier3[-1].weight)

        self.binary_classifier4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier4[-1].weight)

        self.binary_classifier5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier5[-1].weight)

    def forward(self, x):
        x = self.scaling_layer(x)
        features1 = self.slice1(x)
        features2 = self.slice2(features1)
        features3 = self.slice3(features2)
        features4 = self.slice4(features3)
        features5 = self.slice5(features4)

        # torch.Size([1, 64, 256, 256]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32]) torch.Size([1, 512, 16, 16])

        bc1 = self.binary_classifier1(features1).flatten(1)
        bc2 = self.binary_classifier2(features2).flatten(1)
        bc3 = self.binary_classifier3(features3).flatten(1)
        bc4 = self.binary_classifier4(features4).flatten(1)
        bc5 = self.binary_classifier5(features5).flatten(1)

        return bc1 + bc2 + bc3 + bc4 + bc5


from einops import rearrange


def vid2batch(x, b=None):
    return rearrange(x, "b c t h w -> (b t) c h w")


def batch2vid(x, b):
    return rearrange(x, "(b t) c h w -> b c t h w", b=b)


class ThreeDPatchDiscriminator(nn.Module):
    def __init__(self):
        super(ThreeDPatchDiscriminator, self).__init__()
        self.scaling_layer = ScalingLayer()

        _vgg = models.vgg16(pretrained=True)

        self.slice1 = nn.Sequential(_vgg.features[:4])

        self.slice_1_t = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)

        self.slice2 = nn.Sequential(_vgg.features[4:9])
        self.slice_2_t = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)

        self.slice3 = nn.Sequential(_vgg.features[9:16])
        self.slice_3_t = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.slice4 = nn.Sequential(_vgg.features[16:23])
        self.slice_4_t = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)

        self.slice5 = nn.Sequential(_vgg.features[23:30])
        self.slice_5_t = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)

        self.binary_classifier1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier1[-1].weight)

        self.binary_classifier2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier2[-1].weight)

        self.binary_classifier3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv3d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier3[-1].weight)

        self.binary_classifier4 = nn.Sequential(
            nn.Conv3d(512, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier4[-1].weight)

        self.binary_classifier5 = nn.Sequential(
            nn.Conv3d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier5[-1].weight)

        # initialize all the conv3d layers with zero
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.compile
    def forward(self, x):
        batch_size = x.shape[0]
        x = vid2batch(x)

        x = self.scaling_layer(x)
        features1 = self.slice1(x)
        features1 = batch2vid(features1, batch_size)
        # print(features1.shape)
        features1 = features1 + self.slice_1_t(features1).relu()
        features2 = vid2batch(features1)

        features2 = self.slice2(features2)
        features2 = batch2vid(features2, batch_size)
        features2 = features2 + self.slice_2_t(features2).relu()
        features3 = vid2batch(features2)

        features3 = self.slice3(features3)
        features3 = batch2vid(features3, batch_size)
        features3 = features3 + self.slice_3_t(features3).relu()
        # downpool by 2, t axis.
        features3 = features3[:, :, ::2, :, :]
        features4 = vid2batch(features3)

        features4 = self.slice4(features4)
        features4 = batch2vid(features4, batch_size)
        features4 = features4 + self.slice_4_t(features4).relu()

        features5 = vid2batch(features4)

        features5 = self.slice5(features5)
        features5 = batch2vid(features5, batch_size)
        features5 = features5 + self.slice_5_t(features5).relu()
        # features5 = vid2batch(features5)
        # torch.Size([1, 64, 256, 256]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32]) torch.Size([1, 512, 16, 16])
        # print(features1.shape, features2.shape, features3.shape, features4.shape, features5.shape)
        bc1 = self.binary_classifier1(features1)
        bc2 = self.binary_classifier2(features2)
        bc3 = self.binary_classifier3(features3)
        bc4 = self.binary_classifier4(features4)
        bc5 = self.binary_classifier5(features5)

        # print(bc1.shape, bc2.shape, bc3.shape, bc4.shape, bc5.shape)

        bc1 = bc1.flatten(1)
        bc2 = bc2.flatten(1)
        bc3 = bc3.flatten(1)
        bc4 = bc4.flatten(1)
        bc5 = bc5.flatten(1)

        return torch.cat((bc1, bc2, bc3, bc4, bc5), dim=1)


dec_lo, dec_hi = (
    torch.Tensor([-0.1768, 0.3536, 1.0607, 0.3536, -0.1768, 0.0000]),
    torch.Tensor([0.0000, -0.0000, 0.3536, -0.7071, 0.3536, -0.0000]),
)

filters = torch.stack(
    [
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
    ],
    dim=0,
)

filters_expanded = filters.unsqueeze(1)


def prepare_filter(device):
    global filters_expanded
    filters_expanded = filters_expanded.to(device)


def wavelet_transform_multi_channel(x, levels=4):
    B, C, H, W = x.shape
    padded = torch.nn.functional.pad(x, (2, 2, 2, 2))

    # use predefined filters
    global filters_expanded

    ress = []
    for ch in range(C):
        res = torch.nn.functional.conv2d(
            padded[:, ch : ch + 1], filters_expanded, stride=2
        )
        ress.append(res)

    res = torch.cat(ress, dim=1)
    H_out, W_out = res.shape[2], res.shape[3]
    res = res.view(B, C, 4, H_out, W_out)
    res = res.view(B, 4 * C, H_out, W_out)
    return res


import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_video


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dirs,
        t_size=16,
        h_size=256,
        w_size=256,
        split="train",
        frac=0.01,
        seed=0,
    ):
        self.root_dirs = root_dirs
        self.video_files = []
        self.t_size = t_size
        self.h_size = h_size
        self.w_size = w_size

        self.split = split
        self.frac = frac
        self.seed = seed

        for subdir_path in root_dirs:
            if os.path.isdir(subdir_path):
                for video_file in os.listdir(subdir_path):
                    if video_file.endswith(".mp4"):
                        self.video_files.append(os.path.join(subdir_path, video_file))

        # sort, shuffle, set seed, and select frac
        self.video_files.sort()

        random.seed(self.seed)
        random.shuffle(self.video_files)

        if self.split == "val":
            self.video_files = self.video_files[
                : int(len(self.video_files) * self.frac)
            ]
        elif self.split == "train":
            self.video_files = self.video_files[
                int(len(self.video_files) * self.frac) :
            ]

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
            ]
        )

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video, _, _ = read_video(video_path, pts_unit="sec", end_pts=5)  # [T, H, W, C]
        video = video.permute(3, 0, 1, 2)  # [C, T, H, W]
        # normalize to [-1, 1]

        # Random crop in time axis
        if video.shape[1] > self.t_size:
            start_idx = random.randint(0, video.shape[1] - self.t_size)
            video = video[:, start_idx : start_idx + self.t_size, :, :]
        else:
            video = F.pad(video, (0, 0, 0, 0, 0, self.t_size - video.shape[1]))

        video = video / 127.5 - 1.0
        # print(video.shape)
        # Resize spatial dimensions
        video = F.interpolate(
            video.float(),
            size=(self.h_size, self.w_size),
            mode="bilinear",
            align_corners=False,
        )

        # Apply transforms
        video = self.transform(video)

        return video


import imageio
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def create_video_dataloader(
    root_dir,
    batch_size,
    num_workers,
    t_size=16,
    h_size=256,
    w_size=256,
    split="train",
    frac=0.01,
    seed=0,
):

    dataset = VideoDataset(root_dir, t_size, h_size, w_size, split, frac, seed)

    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=42,
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=15,
        sampler=sampler,
    )
    return dataloader


def batch_to_mp4(batch, output_path, fps=30):

    video = batch[0]

    video = video.cpu().numpy()

    video = np.transpose(video, (1, 2, 3, 0))

    video = ((video + 1) * 127.5).astype(np.uint8)

    writer = imageio.get_writer(output_path, fps=fps)

    for frame in video:
        writer.append_data(frame)

    writer.close()

    print(f"Video saved to {output_path}")


def test_patch_discriminator():
    vggDiscriminator = PatchDiscriminator().cuda()
    x = vggDiscriminator(torch.randn(1, 3, 256, 256).cuda())
    print(x.shape)


def test_three_d_patch_discriminator():
    threeDvggDiscriminator = ThreeDPatchDiscriminator().cuda()
    x = threeDvggDiscriminator(torch.randn(7, 3, 64, 256, 256).cuda())
    print(x.shape)


if __name__ == "__main__":
    # vggDiscriminator = PatchDiscriminator().cuda()
    # x = vggDiscriminator(torch.randn(1, 3, 256, 256).cuda())
    # print(x.shape)

    test_three_d_patch_discriminator()

    # dataloader_train = create_video_dataloader(["/home/ubuntu/auravasa/cvhq/35666/"], 8, num_workers=20, t_size = 64, h_size=256, w_size=256, split="train", frac=0.01, seed=0)
    # dataloader_val = create_video_dataloader(["/home/ubuntu/auravasa/cvhq/35666/"], 8, num_workers=20, t_size = 64, h_size=256, w_size=256, split="val", frac=0.01, seed=0)
    # from tqdm import tqdm
    # for i, b in enumerate(tqdm(dataloader_train)):
    #     print(b.shape, b[:2, :2, :2, :2, :2])
    #     if i == 0:
    #         batch_to_mp4(b, f"test_video_{i}.mp4")
