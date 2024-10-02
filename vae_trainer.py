import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import webdataset as wds
import torchvision.transforms as transforms
import torchvision.models as models
import wandb

from utils import LPIPS
import torch
import torch.nn as nn
import torch
import torch.nn as nn

import os
import logging


class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):

        grad_output_norm = torch.norm(grad_output)

        grad_output_normalized = grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized


def gradnorm(x):
    return GradNormFunction.apply(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(groups, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, width_mult=1.0, groups=8):
        super(ResNetEncoder, self).__init__()
        base_channels = int(64 * width_mult)

        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = ResBlock(
            base_channels, base_channels * 2, stride=2, groups=groups
        )

        self.final_conv = nn.Conv2d(
            base_channels * 2, latent_dim, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.final_conv(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3, width_mult=1.0, groups=8):
        super(ResNetDecoder, self).__init__()
        base_channels = int(64 * width_mult)

        self.initial_conv = nn.ConvTranspose2d(
            latent_dim, base_channels * 2, kernel_size=1, stride=1, bias=False
        )

        self.layer1 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels * 2, stride=1, groups=groups),
            nn.ConvTranspose2d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            ResBlock(base_channels, base_channels, stride=1, groups=groups),
            nn.ConvTranspose2d(
                base_channels,
                out_channels,
                kernel_size=4,
                stride=4,
                padding=0,
            ),
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return torch.tanh(x)


class VAE(nn.Module):
    def __init__(self, width_mult=1.0, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = ResNetEncoder(width_mult=width_mult, latent_dim=latent_dim)
        self.decoder = ResNetDecoder(width_mult=width_mult, latent_dim=latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.discriminator(x)


def gan_disc_loss(real_preds, fake_preds):
    real_loss = nn.functional.binary_cross_entropy_with_logits(
        real_preds, torch.ones_like(real_preds)
    )
    fake_loss = nn.functional.binary_cross_entropy_with_logits(
        fake_preds, torch.zeros_like(fake_preds)
    )
    return real_loss + fake_loss


def perceptual_loss(reconstructed, real, vgg_model):
    real_features = vgg_model(real)
    reconstructed_features = vgg_model(reconstructed)
    loss = nn.functional.mse_loss(reconstructed_features, real_features)
    return loss


this_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.CenterCrop(512),
    ]
)


def create_dataloader(url, batch_size, num_workers, do_shuffle=True):
    dataset = wds.WebDataset(
        url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker
    )
    dataset = dataset.shuffle(1000) if do_shuffle else dataset

    dataset = dataset.decode("rgb").to_tuple("jpg;png").map_tuple(this_transform)

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def cleanup():
    dist.destroy_process_group()


def train_ddp(
    dataset_url, test_dataset_url, num_epochs=50, batch_size=16, do_ganloss=True
):

    assert torch.cuda.is_available(), "CUDA is required for DDP"

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    print(f"using device: {device}")

    if master_process:
        wandb.init(project="vae-gan-ddp", entity="simo", name=f"run")

    vae = VAE(width_mult=1.0).cuda()
    discriminator = PatchDiscriminator().cuda()

    vae = DDP(vae, device_ids=[ddp_rank])
    discriminator = DDP(discriminator, device_ids=[ddp_rank])

    optimizer_G = optim.Adam(vae.parameters(), lr=2e-5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-5)

    lpips = LPIPS().cuda()

    dataloader = create_dataloader(
        dataset_url, batch_size, num_workers=4, do_shuffle=True
    )
    test_dataloader = create_dataloader(
        test_dataset_url, batch_size, num_workers=4, do_shuffle=False
    )

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if master_process:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images[0].to(device)
            reconstructed, _ = vae(real_images)

            if do_ganloss:
                real_preds = discriminator(real_images)
                fake_preds = discriminator(reconstructed.detach())
                d_loss = gan_disc_loss(real_preds, fake_preds).mean()
                d_loss.backward(retain_graph=True)

            recon_for_perceptual = gradnorm(reconstructed)
            percep_rec_loss = lpips(recon_for_perceptual, real_images).mean()

            recon_for_mse = gradnorm(reconstructed)
            mse_recon_loss = nn.functional.mse_loss(recon_for_mse, real_images)
            # gan loss
            if do_ganloss:
                recon_for_gan = gradnorm(reconstructed)
                fake_preds = discriminator(recon_for_gan)
                fake_criterion = nn.functional.binary_cross_entropy_with_logits(
                    fake_preds, torch.ones_like(fake_preds)
                )
                gan_loss_value = fake_criterion.mean()

                overall_vae_loss = percep_rec_loss + gan_loss_value + mse_recon_loss
            else:
                overall_vae_loss = percep_rec_loss + mse_recon_loss

            overall_vae_loss.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()

            if do_ganloss:
                optimizer_D.step()
                optimizer_D.zero_grad()

            if master_process:
                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": i,
                        "d_loss": d_loss.item(),
                        "overall_vae_loss": overall_vae_loss.item(),
                        "mse_loss": mse_recon_loss.item(),
                        "perceptual_loss": percep_rec_loss.item(),
                        "gan_loss": gan_loss_value.item(),
                    }
                )

                logger.info(
                    f"Epoch [{epoch}/{num_epochs}] - d_loss: {d_loss.item():.4f}, gan_loss: {gan_loss_value.item():.4f}, perceptual_loss: {percep_rec_loss.item():.4f}, mse_loss: {mse_recon_loss.item():.4f}, overall_vae_loss: {overall_vae_loss.item():.4f}"
                )

        with torch.no_grad():
            if master_process:
                for test_images in test_dataloader:
                    test_images = test_images[0].to(device)
                    reconstructed_test, _ = vae(test_images)

                    # unnormalize the images
                    test_images = test_images * 0.5 + 0.5
                    reconstructed_test = reconstructed_test * 0.5 + 0.5

                    logger.info(f"Epoch [{epoch}/{num_epochs}] - Logging test images")

                    wandb.log(
                        {
                            "reconstructed_test_images": [
                                wandb.Image(test_images[0]),
                                wandb.Image(reconstructed_test[0]),
                                wandb.Image(test_images[1]),
                                wandb.Image(reconstructed_test[1]),
                                wandb.Image(test_images[2]),
                                wandb.Image(reconstructed_test[2]),
                                wandb.Image(test_images[3]),
                                wandb.Image(reconstructed_test[3]),
                            ]
                        }
                    )

                    break

    cleanup()


if __name__ == "__main__":

    dataset_url = "/home/ubuntu/ultimate_pipe/flux_ipadapter_trainer/dataset/art_webdataset/{00000..00127}.tar"
    test_dataset_url = "/home/ubuntu/ultimate_pipe/flux_ipadapter_trainer/dataset/art_webdataset/{00128..00255}.tar"

    train_ddp(dataset_url, test_dataset_url)
