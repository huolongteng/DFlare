from pathlib import Path

import torch
import torch.nn as nn


GAN_PATHS = {
    "mnist": "./models_/GAN/MNIST/dcgan_netG_epoch_99.pth",
    "cifar": "./models_/GAN/CIFAR-10/dcgan_netG_epoch_199.pth",
}


class DCGANGenerator(nn.Module):
    def __init__(self, nc: int, nz: int = 100, ngf: int = 64, mnist: bool = False):
        super().__init__()
        final_padding = 2 if mnist else 0
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=final_padding, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        if z.ndim == 2:
            z = z[:, :, None, None]
        return self.main(z)


def load_gan_generator(dataset: str, device: torch.device, models_dir: str = "./models_/GAN"):
    dataset_dir = "MNIST" if dataset == "mnist" else "CIFAR-10"
    default_name = "dcgan_netG_epoch_99.pth" if dataset == "mnist" else "dcgan_netG_epoch_199.pth"
    path = Path(models_dir) / dataset_dir / default_name
    if not path.exists():
        path = Path(GAN_PATHS[dataset])
    if not path.exists():
        raise FileNotFoundError(f"GAN generator checkpoint not found: {path}")

    if dataset == "mnist":
        model = DCGANGenerator(nc=1, mnist=True)
    elif dataset == "cifar":
        model = DCGANGenerator(nc=3, mnist=False)
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, path


def gan_output_to_numpy(dataset: str, generated: torch.Tensor):
    generated = (generated.detach().float().cpu() + 1.0) / 2.0
    generated = generated.clamp(0.0, 1.0)
    if dataset == "mnist":
        return generated.permute(0, 2, 3, 1).numpy()
    return generated.permute(0, 2, 3, 1).numpy()
