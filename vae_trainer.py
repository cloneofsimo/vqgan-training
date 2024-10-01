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

import torch
import torch.nn as nn
import torch
import torch.nn as nn

import os

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
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
                in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False
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
            ResBlock(
                base_channels * 2, base_channels * 2, stride=1, groups=groups
            ),
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
    def __init__(self, width_mult=1.0):
        super(VAE, self).__init__()
        self.encoder = ResNetEncoder(width_mult=width_mult)
        self.decoder = ResNetDecoder(width_mult=width_mult)

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
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.discriminator(x)

def gan_loss(real_preds, fake_preds):
    real_loss = nn.functional.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss

def perceptual_loss(reconstructed, real, vgg_model):
    real_features = vgg_model(real)
    reconstructed_features = vgg_model(reconstructed)
    loss = nn.functional.mse_loss(reconstructed_features, real_features)
    return loss


this_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(512)
])


def create_dataloader(url, batch_size, num_workers, world_size, rank):
    dataset = (
        wds.WebDataset(url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker)
        .shuffle(1000)
        .decode("rgb")
        .to_tuple("jpg;png")
        .map_tuple(this_transform)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


def cleanup():
    dist.destroy_process_group()

import logging

def train_ddp(world_size, dataset_url, test_dataset_url, num_epochs=50, batch_size=16):
    
    assert torch.cuda.is_available(), "CUDA is required for DDP"
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
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

    optimizer_G = optim.Adam(vae.parameters(), lr=2e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4)

    vgg_model = models.vgg16(pretrained=True).features[:9].to(device)
    vgg_model.eval()

    dataloader = create_dataloader(dataset_url, batch_size, num_workers=4, world_size=world_size, rank=ddp_rank)
    test_dataloader = create_dataloader(test_dataset_url, batch_size, num_workers=4, world_size=world_size, rank=ddp_rank)

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if master_process:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images[0].to(device)
            
            reconstructed, latent = vae(real_images)

            real_preds = discriminator(real_images)
            logger.info(f"real_preds: {real_preds.shape}")
            
            fake_preds = discriminator(reconstructed.detach())
            logger.info(f"fake_preds: {fake_preds.shape}")
            d_loss = gan_loss(real_preds, fake_preds).mean()

            logger.info(f"d_loss: {d_loss}")
            print(f"Rank {ddp_rank} - d_loss: {d_loss}")
            
            

            
            rec_loss_value = perceptual_loss(reconstructed, real_images, vgg_model)
            
            # gan loss
            fake_preds = discriminator(reconstructed)
            fake_criterion = nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))
            gan_loss_value = fake_criterion.mean()
            
            logger.info(f"rec_loss_value: {rec_loss_value}")
            lambda_val = rec_loss_value / (gan_loss_value + 1e-8)
            logger.info(f"lambda_val: {lambda_val}")
            g_loss = rec_loss_value + lambda_val * d_loss

            d_loss.backward(retain_graph=True)
            g_loss.backward()
            
            
            optimizer_G.step()
            optimizer_D.step()
            logger.info(f"optimizer_D.step()")
            
            
            
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            

            if master_process:
                wandb.log({
                    "epoch": epoch,
                    "batch": i,
                    "d_loss": d_loss.item(),
                    "gan_loss": gan_loss_value.item(),
                    "rec_loss": rec_loss_value.item(),
                    "lambda": lambda_val.item()
                })

        
        with torch.no_grad():
            for test_images, _ in test_dataloader:
                test_images = test_images.to(device)
                reconstructed_test, _ = vae(test_images)
                if master_process:
                    logger.info(f"Epoch [{epoch}/{num_epochs}] - Logging test images")
                    wandb.log({"reconstructed_test_images": [wandb.Image(test_images[0]), wandb.Image(reconstructed_test[0])]})
                break

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    dataset_url = "/home/ubuntu/ultimate_pipe/flux_ipadapter_trainer/dataset/art_webdataset/{00000..00032}.tar"
    test_dataset_url = "/home/ubuntu/ultimate_pipe/flux_ipadapter_trainer/dataset/art_webdataset/00008.tar"

    train_ddp(world_size, dataset_url, test_dataset_url)