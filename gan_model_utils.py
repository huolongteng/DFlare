from pathlib import Path

import torch
import torch.nn as nn


GAN_PATHS = {
    "mnist": "./models_/GAN/MNIST/dcgan_netG_epoch_99.pth",
    "fashion": "./models_/GAN/Fashion-MNIST/dcgan_netG_latest.pth",
    "cifar": "./models_/GAN/CIFAR-10/dcgan_netG_epoch_199.pth",
}


class DCGANGenerator(nn.Module):
    def __init__(self, nc: int, nz: int = 100, ngf: int = 64, mnist: bool = False,
                 image_size: int | None = None):
        super().__init__()
        image_size = 28 if mnist else (image_size or 32)
        self.nz = nz
        layers = [
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
        ]
        if image_size == 64:
            layers.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        else:
            final_padding = 2 if image_size == 28 else 0
            layers.append(nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=final_padding, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, z):
        if z.ndim == 2:
            z = z[:, :, None, None]
        return self.main(z)


def load_gan_generator(dataset: str, device: torch.device, models_dir: str = "./models_/GAN"):
    dataset_dirs = {
        "mnist": "MNIST",
        "fashion": "Fashion-MNIST",
        "cifar": "CIFAR-10",
    }
    default_names = {
        "mnist": "dcgan_netG_epoch_99.pth",
        "fashion": "dcgan_netG_latest.pth",
        "cifar": "dcgan_netG_epoch_199.pth",
    }
    dataset_dir = dataset_dirs[dataset]
    default_name = default_names[dataset]
    path = Path(models_dir) / dataset_dir / default_name
    if not path.exists():
        path = Path(GAN_PATHS[dataset])
    if not path.exists():
        raise FileNotFoundError(f"GAN generator checkpoint not found: {path}")

    if dataset in ("mnist", "fashion"):
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
    if dataset in ("mnist", "fashion"):
        return generated.permute(0, 2, 3, 1).numpy()
    return generated.permute(0, 2, 3, 1).numpy()
