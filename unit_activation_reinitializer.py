import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import math


def compute_activation_std(
    model, dataset, device="cpu", batch_size=32, num_workers=0, layer_names=None
):
    activations = {}
    handles = []

    def save_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name].append(output.detach())

        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            activations[name] = []
            handle = module.register_forward_hook(save_activation(name))
            handles.append(handle)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            _ = model(inputs)
            break

    layer_activation_std = {}
    for name in layer_names:
        try:
            act = torch.cat(activations[name], dim=0)
        except:
            print(activations[name])
            break
        act_std = act.std().item()
        layer_activation_std[name] = act_std

    for handle in handles:
        handle.remove()

    return layer_activation_std


def adjust_weight_init(
    model,
    dataset,
    device="cpu",
    batch_size=32,
    num_workers=0,
    tol=0.2,
    max_iters=10,
    exclude_layers=None,
):
    if exclude_layers is None:
        exclude_layers = []

    layers_to_adjust = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and not isinstance(
            module, tuple(exclude_layers)
        ):
            layers_to_adjust.append((name, module))

    print(f"Layers to adjust: {layers_to_adjust}")
    initial_std = {}
    layer_weight_std = {}

    for name, module in layers_to_adjust:
        print(f"Adjusting layer: {name}")
        initial_std[name] = module.weight.std().item()
        fan_in = np.prod(module.weight.shape[1:])
        weight_std = np.sqrt(1 / fan_in)  # use muP for initialization.

        for i in range(max_iters):
            nn.init.normal_(module.weight, std=weight_std)

            activation_std = compute_activation_std(
                model, dataset, device, batch_size, num_workers, layer_names=[name]
            )[name]
            print(f"Iteration {i+1}: Activation std = {activation_std:.4f}")

            if abs(activation_std - 1.0) < tol:
                print(
                    f"Layer {name} achieved near unit activation of {activation_std:.4f} with weight std = {weight_std:.4f}"
                )
                layer_weight_std[name] = weight_std / activation_std
                break
            else:
                weight_std = weight_std / activation_std
        else:
            print(f"Layer {name} did not converge within {max_iters} iterations.")
            layer_weight_std[name] = weight_std

    return initial_std, layer_weight_std


#### HOW TO USE
# 1. define dataset
# 2. define model
# 3. launch.

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    root="mnist_data", train=True, transform=transform, download=True
)


class CustomActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out.mean(dim=[-1, -2])
        return out


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.block1 = ResBlock(1, 256)

        self.fc1 = nn.Linear(256, 256)
        self.act1 = CustomActivation()
        self.ln1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.ln2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, 64)
        self.act3 = nn.Tanh()

        self.fc_residual = nn.Linear(256, 64)

        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        out1 = self.block1(x)
        out1 = out1.view(out1.shape[0], -1)
        out1 = self.act1(out1)
        out1 = self.fc1(out1)
        out1 = self.ln1(out1)

        out2 = self.act2(self.fc2(out1))
        out2 = self.ln2(out2)

        out3 = self.act3(self.fc3(out2))

        res = self.fc_residual(out1)
        out3 += res

        logits = self.fc4(out3)
        return logits


model = MLPModel()


exclude_layers = [nn.LayerNorm]

initial_std, layer_weight_std = adjust_weight_init(
    model,
    dataset=train_dataset,
    device="cuda:0",
    batch_size=64,
    num_workers=0,
    tol=0.1,
    max_iters=10,
    exclude_layers=exclude_layers,
)

print("\nAdjusted Weight Standard Deviations. Before -> After:")
for layer_name, std in layer_weight_std.items():
    print(
        f"Layer {layer_name}, Changed STD from \n   {initial_std[layer_name]:.4f} -> STD {std:.4f}\n"
    )

print(layer_weight_std)
