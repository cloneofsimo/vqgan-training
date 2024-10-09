import os

import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from vae_trainer import (
    VAE,
    PatchDiscriminator,
    create_dataloader,
    gan_loss,
    perceptual_loss,
)


# Sample Image for Testing
def create_sample_image(size=(128, 128), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


# Test the VAE architecture
def test_vae():
    print("Testing VAE Architecture...")
    # Instantiate the model
    vae = VAE(width_mult=1.0)

    # Create a sample image
    img = create_sample_image()

    # Forward pass through the model
    reconstructed, latent = vae(img)

    # Check output shapes
    print(f"Input shape: {img.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")


# Test the Patch Discriminator
def test_discriminator():
    print("Testing Patch Discriminator...")
    # Instantiate the discriminator
    discriminator = PatchDiscriminator()

    # Create a sample image
    img = create_sample_image()

    # Forward pass through the discriminator
    output = discriminator(img)

    # Check output shape
    print(f"Input shape: {img.shape}")
    print(f"Discriminator output shape: {output.shape}")


# Test GAN loss function
def test_gan_loss():
    print("Testing GAN Loss...")
    # Create sample real and fake predictions
    real_preds = torch.randn(1, 1, 8, 8).abs()
    fake_preds = torch.randn(1, 1, 8, 8).abs()

    # Calculate loss
    loss = gan_loss(real_preds, fake_preds)
    print(f"GAN Loss: {loss.item()}")


# Test perceptual loss function
def test_perceptual_loss():
    print("Testing Perceptual Loss...")
    # Instantiate VGG model for perceptual loss
    vgg_model = models.vgg16(pretrained=True).features[:9]

    # Create two sample images (real and reconstructed)
    real_img = create_sample_image()
    reconstructed_img = create_sample_image(color=(0, 255, 0))

    # Calculate perceptual loss
    loss = perceptual_loss(reconstructed_img, real_img, vgg_model)
    print(f"Perceptual Loss: {loss.item()}")


# Test WebDataset DataLoader
def test_webdataset_dataloader():
    print("Testing WebDataset DataLoader...")

    # Dummy URL for WebDataset (replace with actual path or URL in practice)
    dataset_url = "path/to/your/webdataset/shards"

    # Create a dummy dataset with ToTensor transforms
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    # Set up dummy dataloader
    dataloader = create_dataloader(
        dataset_url, batch_size=2, num_workers=4, world_size=1, rank=0
    )

    # Check the output from the dataloader
    for imgs, labels in dataloader:
        print(f"Batch of images shape: {imgs.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break


def test_train_loop():
    print("Testing Train Loop...")

    vae = VAE(width_mult=1.0)
    discriminator = PatchDiscriminator()

    img = create_sample_image()

    optimizer_G = optim.Adam(vae.parameters(), lr=2e-4)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4)

    vgg_model = models.vgg16(pretrained=True).features[:9]
    vgg_model.eval()

    reconstructed, latent = vae(img)

    print(f"Input image shape: {img.shape} (Expected: [1, 3, 128, 128])")
    print(f"Latent shape: {latent.shape} (Expected: [1, 256, 8, 8])")
    print(
        f"Reconstructed image shape: {reconstructed.shape} (Expected: [1, 3, 128, 128])"
    )

    real_preds = discriminator(img)
    fake_preds = discriminator(reconstructed.detach())

    print(f"Real predictions shape: {real_preds.shape} (Expected: [1, 1, 8, 8])")
    print(f"Fake predictions shape: {fake_preds.shape} (Expected: [1, 1, 8, 8])")

    gan_loss_value = gan_loss(real_preds, fake_preds)
    rec_loss_value = perceptual_loss(reconstructed, img, vgg_model)

    lambda_val = rec_loss_value / (gan_loss_value + 1e-8)
    g_loss = rec_loss_value + lambda_val * gan_loss_value
    g_loss.backward(retain_graph=True)

    d_loss = gan_loss(real_preds, fake_preds)
    d_loss.backward()

    optimizer_D.step()
    optimizer_G.step()

    optimizer_G.zero_grad()
    optimizer_D.zero_grad()

    print(f"Discriminator Loss: {d_loss.item()} (Expected: > 0)")
    print(f"Generator Loss: {g_loss.item()} (Expected: > 0)")
    print(
        f"Lambda value (weight): {lambda_val.item()} (Expected: ~1.0 depending on losses)"
    )


# Run the dummy train loop test
if __name__ == "__main__":

    # download lpips from here
    # https://heibox.uni-heidelberg.de/seafhttp/files/9535cbee-6558-4c0c-8743-78f5e56ea75e/vgg.pth
    os.system(
        "wget https://heibox.uni-heidelberg.de/seafhttp/files/9535cbee-6558-4c0c-8743-78f5e56ea75e/vgg.pth"
    )
    test_vae()
    test_discriminator()
    test_gan_loss()
    test_perceptual_loss()
    # test_webdataset_dataloader()

    # Dummy train loop for input-output
    test_train_loop()
