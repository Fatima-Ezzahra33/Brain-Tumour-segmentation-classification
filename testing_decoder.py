import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import os


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NOISE_STD = 0.15
IMG_SIZE = 128
BATCH_SIZE = 32

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ============================================================
# DATASET
# ============================================================
class NoisyMRIDataset(Dataset):
    def __init__(self, base_dataset, indices, transform, noise_std=0.15):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.noise_std = noise_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_pil, _ = self.base_dataset[self.indices[idx]]
        clean = self.transform(img_pil)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, -1.0, 1.0)
        return noisy, clean

# ============================================================
# MODEL
# ============================================================
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 2, stride=2),
        )
        self.out_act = nn.Tanh()

    def forward(self, x):
        return self.out_act(self.decoder(self.encoder(x)))

# ============================================================
# VISUALIZATION
# ============================================================
def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

def show_denoising_results(model, loader, device, n=6, save_path='denoising_results.png'):
    model.eval()
    noisy_imgs, clean_imgs, recon_imgs = [], [], []

    with torch.no_grad():
        for noisy, clean in loader:
            recon = model(noisy.to(device)).cpu()
            noisy_imgs.append(noisy)
            clean_imgs.append(clean)
            recon_imgs.append(recon)
            if sum(x.size(0) for x in noisy_imgs) >= n:
                break

    noisy_imgs = torch.cat(noisy_imgs)[:n]
    clean_imgs = torch.cat(clean_imgs)[:n]
    recon_imgs = torch.cat(recon_imgs)[:n]

    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 8))
    rows = [noisy_imgs, recon_imgs, clean_imgs]
    titles = ['Noisy Input', 'Denoised Output', 'Clean Reference']

    for i, (row_imgs, title) in enumerate(zip(rows, titles)):
        axes[i, 0].set_ylabel(title, fontsize=11, fontweight='bold')
        for j in range(n):
            img = denorm(row_imgs[j]).permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

    plt.suptitle('Autoencoder Denoising — MRI Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to '{save_path}'")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Loading model...")
    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load('autoencoder.pth', map_location=DEVICE))
    model.eval()

    print("Loading dataset...")
    source = datasets.ImageFolder('data/Training', transform=None)
    val_size = int(0.2 * len(source))
    train_size = len(source) - val_size
    train_idx, val_idx = random_split(
        range(len(source)), [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    val_ds = NoisyMRIDataset(source, val_idx, base_transform, noise_std=NOISE_STD)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Generating visualization ({len(val_ds)} val images)...")
    show_denoising_results(model, val_loader, DEVICE, n=6)