---
license: apache-2.0
---

# Equivariant 16ch, f8 VAE

<video controls autoplay src="https://cdn-uploads.huggingface.co/production/uploads/6311151c64939fabc00c8436/6DQGRWvQvDXp2xQlvwvwU.mp4"></video>

AuraEquiVAE is a novel autoencoder that addresses multiple problems of existing conventional VAEs. First, unlike traditional VAEs that have significantly small log-variance, this model admits large noise to the latent space.
Additionally, unlike traditional VAEs, the latent space is equivariant under `Z_2 X Z_2` group operations (Horizontal / Vertical flip).

To understand the equivariance, we apply suitable group actions to both the latent space globally and locally. The latent is represented as `Z = (z_1, ..., z_n)`, and we perform a global permutation group action `g_global` on the tuples such that `g_global` is isomorphic to the `Z_2 x Z_2` group.
We also apply a local action `g_local` to individual `z_i` elements such that `g_local` is also isomorphic to the `Z_2 x Z_2` group.

In our specific case, `g_global` corresponds to flips, while `g_local` corresponds to sign flips on specific latent dimensions. Changing 2 channels for sign flips for both horizontal and vertical directions was chosen empirically.

The model has been trained using the approach described in [Mastering VAE Training](https://github.com/cloneofsimo/vqgan-training), where detailed explanations for the training process can be found.

## How to use

To use the weights, copy paste the [VAE](https://github.com/cloneofsimo/vqgan-training/blob/03e04401cf49fe55be612d1f568be0110aa0fad1/ae.py) implementation.

```python
from ae import VAE
import torch
from PIL import Image

vae = VAE(
    resolution=256,
    in_channels=3,
    ch=256,
    out_ch=3,
    ch_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    z_ch
).cuda().bfloat16()

from safetensors.torch import load_file
state_dict = load_file("./vae_epoch_3_step_49501_bf16.pt")
vae.load_state_dict(state_dict)

imgpath = 'contents/lavender.jpg'

img_orig = Image.open(imgpath).convert("RGB")
offset = 128
W = 768
img_orig = img_orig.crop((offset, offset, W + offset, W + offset))
img = transforms.ToTensor()(img_orig).unsqueeze(0).cuda()
img = (img - 0.5) / 0.5

with torch.no_grad():
    z = vae.encoder(img)
    z = z.clamp(-8.0, 8.0) # this is latent!!

# flip horizontal
z = torch.flip(z, [-1]) # this corresponds to g_global
z[:, -4:-2] = -z[:, -4:-2] # this corresponds to g_local

# flip vertical
z = torch.flip(z, [-2])
z[:, -2:] = -z[:, -2:]


with torch.no_grad():
    decz = vae.decoder(z) # this is image!

decimg = ((decz + 1) / 2).clamp(0, 1).squeeze(0).cpu().float().numpy().transpose(1, 2, 0)
decimg = (decimg * 255).astype('uint8')
decimg = Image.fromarray(decimg) # PIL image.

```

## Citation

If you find this model useful, please cite:

```
@misc{Training VQGAN and VAE, with detailed explanation,
  author = {Simo Ryu},
  title = {Training VQGAN and VAE, with detailed explanation},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cloneofsimo/vqgan-training}},
}
```


