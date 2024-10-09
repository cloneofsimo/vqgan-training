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
import torch.nn.functional as F

import numpy as np
import random

import click
import os
import logging

from utils import LPIPS, PatchDiscriminator

from ae import VAE


class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):

        grad_output_norm = torch.linalg.norm(grad_output, dim=0, keepdim=True)
        # print(grad_output_norm.shape)

        grad_output_normalized = grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized


def gradnorm(x):
    return GradNormFunction.apply(x)


def gan_disc_loss(real_preds, fake_preds):
    real_loss = nn.functional.binary_cross_entropy_with_logits(
        real_preds, torch.ones_like(real_preds)
    )
    fake_loss = nn.functional.binary_cross_entropy_with_logits(
        fake_preds, torch.zeros_like(fake_preds)
    )
    # eval its online performance
    got_real_right = real_preds.mean()
    got_fake_right = fake_preds.mean()
    return real_loss + fake_loss, got_real_right, got_fake_right


this_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.CenterCrop(256),
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


class SoftLargePenalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        x_where_large = (x.abs() > threshold).float()
        return grad_output * x_where_large, None


def soft_large_penalty(x, threshold=1.0):
    # return SoftLargePenalty.apply(x, torch.tensor(threshold, dtype=x.dtype, device=x.device))
    return x


def vae_loss_function(x, x_reconstructed, z):
    # downsample images by factor of 8
    x_reconstructed_down = F.interpolate(
        x_reconstructed, scale_factor=1 / 8, mode="bilinear", align_corners=False
    )
    x_down = F.interpolate(x, scale_factor=1 / 8, mode="bilinear", align_corners=False)

    recon_loss = (x_reconstructed_down - x_down).abs().mean()

    # KL loss
    mean, logvar = torch.chunk(z, 2, dim=1)
    elewise_kl_loss = -1 - logvar + logvar.exp()
    modified_kl_loss = soft_large_penalty(elewise_kl_loss, threshold=0.4).pow(2)

    elewise_mean_loss = mean.pow(2)
    modified_mean_loss = soft_large_penalty(elewise_mean_loss, threshold=0.4)

    total_loss = (modified_kl_loss + modified_mean_loss).mean()

    with torch.no_grad():
        actual_kl_loss = elewise_kl_loss.mean()
        actual_mean_loss = elewise_mean_loss.mean()
        actual_ks_loss = (actual_kl_loss + actual_mean_loss).mean()

    vae_loss = recon_loss + total_loss * 4.0
    return vae_loss, {
        "recon_loss": recon_loss.item(),
        "kl_loss": actual_ks_loss.item(),
        "average_of_abs_z": z.abs().mean().item(),
        "std_of_abs_z": z.abs().std().item(),
        "average_of_logvar": logvar.mean().item(),
        "std_of_logvar": logvar.std().item(),
    }


def cleanup():
    dist.destroy_process_group()


@click.command()
@click.option(
    "--dataset_url", type=str, default="", help="URL for the training dataset"
)
@click.option(
    "--test_dataset_url", type=str, default="", help="URL for the test dataset"
)
@click.option("--num_epochs", type=int, default=2, help="Number of training epochs")
@click.option("--batch_size", type=int, default=16, help="Batch size for training")
@click.option("--do_ganloss", is_flag=True, help="Whether to use GAN loss")
@click.option(
    "--learning_rate_vae", type=float, default=1e-5, help="Learning rate for VAE"
)
@click.option(
    "--learning_rate_disc",
    type=float,
    default=2e-4,
    help="Learning rate for discriminator",
)
@click.option("--vae_resolution", type=int, default=256, help="Resolution for VAE")
@click.option("--vae_in_channels", type=int, default=3, help="Input channels for VAE")
@click.option("--vae_ch", type=int, default=256, help="Base channel size for VAE")
@click.option(
    "--vae_ch_mult", type=str, default="1,2,4,4", help="Channel multipliers for VAE"
)
@click.option(
    "--vae_num_res_blocks",
    type=int,
    default=2,
    help="Number of residual blocks for VAE",
)
@click.option(
    "--vae_z_channels", type=int, default=16, help="Number of latent channels for VAE"
)
@click.option(
    '--run_name', type=str, default='run', help='Name of the run for wandb'
)
def train_ddp(
    dataset_url,
    test_dataset_url,
    num_epochs,
    batch_size,
    do_ganloss,
    learning_rate_vae,
    learning_rate_disc,
    vae_resolution,
    vae_in_channels,
    vae_ch,
    vae_ch_mult,
    vae_num_res_blocks,
    vae_z_channels,
    run_name,
):
    
    # fix random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    start_train = 0
    end_train = 128 * 3

    start_test = end_train + 1
    end_test = start_test + 8

    dataset_url = f"/home/ubuntu/ultimate_pipe/flux_ipadapter_trainer/dataset/art_webdataset/{{{start_train:05d}..{end_train:05d}}}.tar"
    test_dataset_url = f"/home/ubuntu/ultimate_pipe/flux_ipadapter_trainer/dataset/art_webdataset/{{{start_test:05d}..{end_test:05d}}}.tar"


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
        wandb.init(project="vae-gan-ddp-sweep", entity="simo", name=run_name,
                   config={
                       "learning_rate_vae": learning_rate_vae,
                       "learning_rate_disc": learning_rate_disc,
                       "vae_ch": vae_ch,
                       "vae_resolution": vae_resolution,
                       "vae_in_channels": vae_in_channels,
                       "vae_ch_mult": vae_ch_mult,
                       "vae_num_res_blocks": vae_num_res_blocks,
                       "vae_z_channels": vae_z_channels,
                       "batch_size": batch_size,
                       "num_epochs": num_epochs,
                       "do_ganloss": do_ganloss,
                   })

    vae = VAE(
        resolution=vae_resolution,
        in_channels=vae_in_channels,
        ch=vae_ch,
        out_ch=vae_in_channels,
        ch_mult=[int(x) for x in vae_ch_mult.split(",")],
        num_res_blocks=vae_num_res_blocks,
        z_channels=vae_z_channels,
    ).cuda()

    discriminator = PatchDiscriminator().cuda()

    vae = DDP(vae, device_ids=[ddp_rank])
    discriminator = DDP(discriminator, device_ids=[ddp_rank])

    optimizer_G = optim.AdamW(vae.parameters(), lr=learning_rate_vae / vae_ch, weight_decay=1e-3, betas=(0.9, 0.95))
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=learning_rate_disc / vae_ch, weight_decay=1e-3, betas=(0.9, 0.95))

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
    global_step = 0
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            
            real_images = real_images[0].to(device)
            reconstructed, z = vae(real_images)

            if do_ganloss:
                real_preds = discriminator(real_images)
                fake_preds = discriminator(reconstructed.detach())
                d_loss = gan_disc_loss(real_preds, fake_preds).mean()
                d_loss.backward(retain_graph=True)

            # unnormalize the images, and perceptual loss
            _recon_for_perceptual = gradnorm(reconstructed)
            _real_images = real_images * 0.5 + 0.5
            _recon_for_perceptual = _recon_for_perceptual * 0.5 + 0.5
            percep_rec_loss = lpips(_recon_for_perceptual, _real_images).mean()

            # mse, vae loss.
            recon_for_mse = gradnorm(reconstructed)
            vae_loss, loss_data = vae_loss_function(real_images, recon_for_mse, z)
            # gan loss
            if do_ganloss:
                recon_for_gan = gradnorm(reconstructed)
                fake_preds = discriminator(recon_for_gan)
                fake_criterion = nn.functional.binary_cross_entropy_with_logits(
                    fake_preds, torch.ones_like(fake_preds)
                )
                gan_loss_value = fake_criterion.mean()

                overall_vae_loss = percep_rec_loss + gan_loss_value + vae_loss
            else:
                overall_vae_loss = percep_rec_loss + vae_loss

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
                        "d_loss": d_loss.item() if do_ganloss else None,
                        "overall_vae_loss": overall_vae_loss.item(),
                        "mse_loss": loss_data["recon_loss"],
                        "kl_loss": loss_data["kl_loss"],
                        "perceptual_loss": percep_rec_loss.item(),
                        "gan_loss": gan_loss_value.item() if do_ganloss else None,
                        "abs_mu": loss_data["average_of_abs_z"],
                        "abs_logvar": loss_data["average_of_logvar"]
                    }
                )
                
                if global_step % 200 == 0:
         
                    wandb.log({
                        f"loss_stepwise/mse_loss_{global_step}": loss_data["recon_loss"],
                        f"loss_stepwise/kl_loss_{global_step}": loss_data["kl_loss"],
                        f"loss_stepwise/overall_vae_loss_{global_step}": overall_vae_loss.item(),
                    })
                        

                log_message = f"Epoch [{epoch}/{num_epochs}] - "
                log_items = [
                    ("perceptual_loss", percep_rec_loss.item()),
                    ("mse_loss", loss_data["recon_loss"]),
                    ("kl_loss", loss_data["kl_loss"]),
                    ("overall_vae_loss", overall_vae_loss.item()),
                    ("ABS mu (0.0): average_of_abs_z", loss_data["average_of_abs_z"]),
                    ("STD mu : std_of_abs_z", loss_data["std_of_abs_z"]),
                    (
                        "ABS logvar (0.0) : average_of_logvar",
                        loss_data["average_of_logvar"],
                    ),
                    ("STD logvar : std_of_logvar", loss_data["std_of_logvar"]),
                ]

                if do_ganloss:
                    log_items = [
                        ("d_loss", d_loss.item()),
                        ("gan_loss", gan_loss_value.item()),
                    ] + log_items

                log_message += "\n\t".join(
                    [f"{key}: {value:.4f}" for key, value in log_items]
                )
                logger.info(log_message)
                
                global_step += 1

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
                
            os.makedirs(f"./ckpt/{run_name}", exist_ok=True)
            torch.save(vae.state_dict(), f"./ckpt/{run_name}/vae_epoch_{epoch}.pt")
    cleanup()


if __name__ == "__main__":

    #  torchrun --nproc_per_node=8 vae_trainer.py
    train_ddp()

    
    # train_ddp(dataset_url, test_dataset_url)
