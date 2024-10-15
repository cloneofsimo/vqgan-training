import logging
import os
import random

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import GaussianBlur
from transformers import get_cosine_schedule_with_warmup

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import wandb
from ae import VAE
from utils import LPIPS, PatchDiscriminator


class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]

        grad_output_norm = torch.linalg.vector_norm(
            grad_output, dim=list(range(1, len(grad_output.shape))), keepdim=True
        )

        grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized, None


def gradnorm(x, weight=1.0):
    weight = torch.tensor(weight, device=x.device)
    return GradNormFunction.apply(x, weight)


def avg_scalar_over_nodes(value: float, device):
    value = torch.tensor(value, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value.item()


def gan_disc_loss(real_preds, fake_preds):
    real_loss = nn.functional.binary_cross_entropy_with_logits(
        real_preds, torch.ones_like(real_preds)
    )
    fake_loss = nn.functional.binary_cross_entropy_with_logits(
        fake_preds, torch.zeros_like(fake_preds)
    )
    # eval its online performance
    avg_real_preds = real_preds.mean().item()
    avg_fake_preds = fake_preds.mean().item()

    return real_loss + fake_loss, avg_real_preds, avg_fake_preds


this_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.CenterCrop(512),
        transforms.Resize(256),
    ]
)


def this_transform_random_crop_resize(x, width=256):

    x = transforms.ToTensor()(x)
    x = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)

    if random.random() < 0.5:
        x = transforms.RandomCrop(width)(x)
    else:
        x = transforms.Resize(width)(x)
        x = transforms.RandomCrop(width)(x)

    return x


def create_dataloader(url, batch_size, num_workers, do_shuffle=True, just_resize=False):
    dataset = wds.WebDataset(
        url, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker
    )
    dataset = dataset.shuffle(1000) if do_shuffle else dataset

    dataset = (
        dataset.decode("rgb")
        .to_tuple("jpg;png")
        .map_tuple(
            this_transform_random_crop_resize if not just_resize else this_transform
        )
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def blurriness_heatmap(input_image):
    grayscale_image = input_image.mean(dim=1, keepdim=True)

    laplacian_kernel = torch.tensor(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, -20, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    laplacian_kernel = laplacian_kernel.view(1, 1, 5, 5)

    laplacian_kernel = laplacian_kernel.to(input_image.device)

    edge_response = F.conv2d(grayscale_image, laplacian_kernel, padding=2)

    edge_magnitude = GaussianBlur(kernel_size=(13, 13), sigma=(2.0, 2.0))(
        edge_response.abs()
    )

    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (
        edge_magnitude.max() - edge_magnitude.min() + 1e-8
    )

    blurriness_map = 1 - edge_magnitude

    blurriness_map = torch.where(
        blurriness_map < 0.8, torch.zeros_like(blurriness_map), blurriness_map
    )

    return blurriness_map.repeat(1, 3, 1, 1)


def vae_loss_function(x, x_reconstructed, z, do_pool=True):
    # downsample images by factor of 8
    if do_pool:
        x_reconstructed_down = F.interpolate(
            x_reconstructed, scale_factor=1 / 16, mode="area"
        )
        x_down = F.interpolate(x, scale_factor=1 / 16, mode="area")
        recon_loss = ((x_reconstructed_down - x_down)).abs().mean()
    else:
        x_reconstructed_down = x_reconstructed
        x_down = x

        recon_loss = (
            ((x_reconstructed_down - x_down) * blurriness_heatmap(x_down)).abs().mean()
        )

    elewise_mean_loss = z.pow(2)
    zloss = elewise_mean_loss.mean()

    with torch.no_grad():
        actual_mean_loss = elewise_mean_loss.mean()
        actual_ks_loss = actual_mean_loss.mean()

    vae_loss = recon_loss * 0.0 + zloss * 1.0
    return vae_loss, {
        "recon_loss": recon_loss.item(),
        "kl_loss": actual_ks_loss.item(),
        "average_of_abs_z": z.abs().mean().item(),
        "std_of_abs_z": z.abs().std().item(),
        "average_of_logvar": 0.0,
        "std_of_logvar": 0.0,
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
@click.option("--batch_size", type=int, default=8, help="Batch size for training")
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
@click.option("--run_name", type=str, default="run", help="Name of the run for wandb")
@click.option(
    "--max_steps", type=int, default=1000, help="Maximum number of steps to train for"
)
@click.option(
    "--evaluate_every_n_steps", type=int, default=250, help="Evaluate every n steps"
)
@click.option("--load_path", type=str, default=None, help="Path to load the model from")
@click.option("--do_clamp", is_flag=True, help="Whether to clamp the latent codes")
@click.option(
    "--clamp_th", type=float, default=8.0, help="Clamp threshold for the latent codes"
)
@click.option(
    "--max_spatial_dim",
    type=int,
    default=256,
    help="Maximum spatial dimension for overall training",
)
@click.option(
    "--do_attn", type=bool, default=False, help="Whether to use attention in the VAE"
)
@click.option(
    "--project_name",
    type=str,
    default="vae_sweep_attn_lr_width",
    help="Project name for wandb",
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
    max_steps,
    evaluate_every_n_steps,
    load_path,
    do_clamp,
    clamp_th,
    max_spatial_dim,
    do_attn,
    project_name,
):

    # fix random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)

    start_train = 0
    end_train = 128 * 16

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
        wandb.init(
            project=project_name,
            entity="simo",
            name=run_name,
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
                "do_attn": do_attn,
            },
        )

    vae = VAE(
        resolution=vae_resolution,
        in_channels=vae_in_channels,
        ch=vae_ch,
        out_ch=vae_in_channels,
        ch_mult=[int(x) for x in vae_ch_mult.split(",")],
        num_res_blocks=vae_num_res_blocks,
        z_channels=vae_z_channels,
        use_attn=do_attn,
    ).cuda()

    discriminator = PatchDiscriminator().cuda()
    discriminator.requires_grad_(True)

    vae = DDP(vae, device_ids=[ddp_rank])

    # vae.module.encoder = torch.compile(
    #     vae.module.encoder, fullgraph=False, mode="reduce-overhead"
    # )
    # vae.module.decoder = torch.compile(
    #     vae.module.decoder, fullgraph=False, mode="reduce-overhead"
    # )

    discriminator = DDP(discriminator, device_ids=[ddp_rank])

    # context
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    optimizer_G = optim.AdamW(
        [
            {
                "params": [p for n, p in vae.named_parameters() if "conv_in" not in n],
                "lr": learning_rate_vae / vae_ch,
            },
            {
                "params": [p for n, p in vae.named_parameters() if "conv_in" in n],
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-3,
        betas=(0.9, 0.95),
    )

    optimizer_D = optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate_disc,
        weight_decay=1e-3,
        betas=(0.9, 0.95),
    )

    lpips = LPIPS().cuda()

    dataloader = create_dataloader(
        dataset_url, batch_size, num_workers=4, do_shuffle=True
    )
    test_dataloader = create_dataloader(
        test_dataset_url, batch_size, num_workers=4, do_shuffle=False, just_resize=True
    )

    num_training_steps = max_steps
    num_warmup_steps = 200
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer_G, num_warmup_steps, num_training_steps
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

    if load_path is not None:
        state_dict = torch.load(load_path, map_location="cpu")
        # state_dict = {
        #     k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        # }
        status = vae.load_state_dict(state_dict, strict=False)
        print(status)

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):

            real_images = real_images[0].to(device)
            z = vae.module.encoder(real_images)

            # z distribution
            with ctx:
                z_dist_value: torch.Tensor = z.detach().cpu().reshape(-1)

            def kurtosis(x):
                return ((x - x.mean()) ** 4).mean() / (x.std() ** 4)

            def skew(x):
                return ((x - x.mean()) ** 3).mean() / (x.std() ** 3)

            z_quantiles = {
                "0.0": z_dist_value.quantile(0.0),
                "0.2": z_dist_value.quantile(0.2),
                "0.4": z_dist_value.quantile(0.4),
                "0.6": z_dist_value.quantile(0.6),
                "0.8": z_dist_value.quantile(0.8),
                "1.0": z_dist_value.quantile(1.0),
                "kurtosis": kurtosis(z_dist_value),
                "skewness": skew(z_dist_value),
            }

            if do_clamp:
                z = z.clamp(-clamp_th, clamp_th)
            z_s = vae.module.reg(z)

            #### do aug

            if random.random() < 0.5:
                z_s = torch.flip(z_s, [-1])
                z_s[:, -4:-2] = -z_s[:, -4:-2]
                real_images = torch.flip(real_images, [-1])

            if random.random() < 0.5:
                z_s = torch.flip(z_s, [-2])
                z_s[:, -2:] = -z_s[:, -2:]
                real_images = torch.flip(real_images, [-2])

            if random.random() < 0.5:
                # crop image and latent.'
                new_z_h = random.randint(8, max_spatial_dim // 8 - 1)
                new_z_w = random.randint(8, max_spatial_dim // 8 - 1)
                offset_z_h = random.randint(0, max_spatial_dim // 8 - new_z_h - 1)
                offset_z_w = random.randint(0, max_spatial_dim // 8 - new_z_w - 1)

                new_h = new_z_h * 8
                new_w = new_z_w * 8
                offset_h = offset_z_h * 8
                offset_w = offset_z_w * 8

                real_images = real_images[
                    :, :, offset_h : offset_h + new_h, offset_w : offset_w + new_w
                ]
                z_s = z_s[
                    :,
                    :,
                    offset_z_h : offset_z_h + new_z_h,
                    offset_z_w : offset_z_w + new_z_w,
                ]

                assert real_images.shape[-2] == new_h
                assert real_images.shape[-1] == new_w
                assert z_s.shape[-2] == new_z_h
                assert z_s.shape[-1] == new_z_w

            with ctx:
                reconstructed = vae.module.decoder(z_s)

            if global_step >= max_steps:
                break

            if do_ganloss:
                real_preds = discriminator(real_images)
                fake_preds = discriminator(reconstructed.detach())
                d_loss, avg_real_logits, avg_fake_logits = gan_disc_loss(
                    real_preds, fake_preds
                )
                d_loss = d_loss.mean()
                d_loss.backward(retain_graph=True)

                avg_real_logits = avg_scalar_over_nodes(avg_real_logits, device)
                avg_fake_logits = avg_scalar_over_nodes(avg_fake_logits, device)

            # unnormalize the images, and perceptual loss
            _recon_for_perceptual = gradnorm(reconstructed)

            percep_rec_loss = lpips(_recon_for_perceptual, real_images).mean()

            # mse, vae loss.
            recon_for_mse = gradnorm(reconstructed, weight=0.001)
            vae_loss, loss_data = vae_loss_function(real_images, recon_for_mse, z)
            # gan loss
            if do_ganloss and global_step >= 20:
                recon_for_gan = gradnorm(reconstructed)
                fake_preds = discriminator(recon_for_gan)
                real_preds_const = real_preds.clone().detach()
                # loss where (real > fake + 0.01)
                g_gan_loss = (real_preds_const - fake_preds - 0.1).relu().mean()

                overall_vae_loss = percep_rec_loss + g_gan_loss + vae_loss
                g_gan_loss = g_gan_loss.item()
            else:
                overall_vae_loss = percep_rec_loss + vae_loss
                g_gan_loss = 0.0

            overall_vae_loss.backward()
            optimizer_G.step()
            optimizer_G.zero_grad()
            lr_scheduler.step()

            if do_ganloss:
                optimizer_D.step()
                optimizer_D.zero_grad()

            if master_process:
                if global_step % 5 == 0:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "batch": i,
                            "gan/discriminator_loss": (
                                d_loss.item() if do_ganloss else None
                            ),
                            "overall_vae_loss": overall_vae_loss.item(),
                            "mse_loss": loss_data["recon_loss"],
                            "kl_loss": loss_data["kl_loss"],
                            "perceptual_loss": percep_rec_loss.item(),
                            "gan/generator_gan_loss": (
                                g_gan_loss if do_ganloss else None
                            ),
                            "z_quantiles/abs_z": loss_data["average_of_abs_z"],
                            "z_quantiles/std_z": loss_data["std_of_abs_z"],
                            "z_quantiles/logvar": loss_data["average_of_logvar"],
                            "gan/avg_real_logits": (
                                avg_real_logits if do_ganloss else None
                            ),
                            "gan/avg_fake_logits": (
                                avg_fake_logits if do_ganloss else None
                            ),
                            "z_quantiles/qs": z_quantiles,
                        }
                    )

                if global_step % 200 == 0:

                    wandb.log(
                        {
                            f"loss_stepwise/mse_loss_{global_step}": loss_data[
                                "recon_loss"
                            ],
                            f"loss_stepwise/kl_loss_{global_step}": loss_data[
                                "kl_loss"
                            ],
                            f"loss_stepwise/overall_vae_loss_{global_step}": overall_vae_loss.item(),
                        }
                    )

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
                    *[(f"z_quantiles/{q}", v) for q, v in z_quantiles.items()],
                ]

                if do_ganloss:
                    log_items = [
                        ("d_loss", d_loss.item()),
                        ("gan_loss", g_gan_loss),
                        ("avg_real_logits", avg_real_logits),
                        ("avg_fake_logits", avg_fake_logits),
                    ] + log_items

                log_message += "\n\t".join(
                    [f"{key}: {value:.4f}" for key, value in log_items]
                )
                logger.info(log_message)

            global_step += 1

            if (
                evaluate_every_n_steps > 0
                and global_step % evaluate_every_n_steps == 1
                and master_process
            ):

                with torch.no_grad():
                    all_test_images = []
                    all_reconstructed_test = []

                    for test_images in test_dataloader:
                        test_images = test_images[0].to(device)
                        z = vae.module.encoder(test_images)

                        if do_clamp:
                            z = z.clamp(-clamp_th, clamp_th)

                        z_s = vae.module.reg(z)

                        # [1, 2]
                        # [3, 4]
                        # ->
                        # [3, 4]
                        # [1, 2]
                        # ->
                        # [4, 3]
                        # [2, 1]

                        z_s = torch.flip(z_s, [-1, -2])
                        z_s[:, -4:] = -z_s[:, -4:]

                        reconstructed_test = vae.module.decoder(z_s)

                        # unnormalize the images
                        test_images = test_images * 0.5 + 0.5
                        reconstructed_test = reconstructed_test * 0.5 + 0.5
                        # clamp
                        test_images = test_images.clamp(0, 1)
                        reconstructed_test = reconstructed_test.clamp(0, 1)

                        # flip twice
                        reconstructed_test = torch.flip(reconstructed_test, [-1, -2])

                        all_test_images.append(test_images)
                        all_reconstructed_test.append(reconstructed_test)

                        if len(all_test_images) >= 2:
                            break

                    test_images = torch.cat(all_test_images, dim=0)
                    reconstructed_test = torch.cat(all_reconstructed_test, dim=0)

                    logger.info(f"Epoch [{epoch}/{num_epochs}] - Logging test images")

                    # crop test and recon to 64 x 64
                    D = 256
                    offset = 0
                    test_images = test_images[
                        :, :, offset : offset + D, offset : offset + D
                    ].cpu()
                    reconstructed_test = reconstructed_test[
                        :, :, offset : offset + D, offset : offset + D
                    ].cpu()

                    # concat the images into one large image.
                    # make size of (D * 4) x (D * 4)
                    recon_all_image = torch.zeros((3, D * 4, D * 4))
                    test_all_image = torch.zeros((3, D * 4, D * 4))

                    for i in range(2):
                        for j in range(4):
                            recon_all_image[
                                :, i * D : (i + 1) * D, j * D : (j + 1) * D
                            ] = reconstructed_test[i * 4 + j]
                            test_all_image[
                                :, i * D : (i + 1) * D, j * D : (j + 1) * D
                            ] = test_images[i * 4 + j]

                    wandb.log(
                        {
                            "reconstructed_test_images": [
                                wandb.Image(recon_all_image),
                            ],
                            "test_images": [
                                wandb.Image(test_all_image),
                            ],
                        }
                    )

                    os.makedirs(f"./ckpt/{run_name}", exist_ok=True)
                    torch.save(
                        vae.state_dict(),
                        f"./ckpt/{run_name}/vae_epoch_{epoch}_step_{global_step}.pt",
                    )
                    print(
                        f"Saved checkpoint to ./ckpt/{run_name}/vae_epoch_{epoch}_step_{global_step}.pt"
                    )

    cleanup()


if __name__ == "__main__":

    # Example: torchrun --nproc_per_node=8 vae_trainer.py
    train_ddp()
