import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import os

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
VAL_SPLIT  = 0.2
SEED       = 42
NUM_EPOCHS = 25
LR         = 0.001
NOISE_STD  = 0.15   # Gaussian noise strength (0.0–1.0 range after normalization)
IMG_SIZE   = 128

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# TRANSFORMS
# ============================================================
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                          # [0, 1]
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])           # [-1, 1]
])

# ============================================================
# DATASET — adds Gaussian noise on-the-fly
# ============================================================
class NoisyMRIDataset(Dataset):
    """
    Wraps an ImageFolder dataset.
    Returns (noisy_image, clean_image) pairs.
    The clean image is the original; the noisy version is
    generated on-the-fly by adding Gaussian noise.
    """
    def __init__(self, base_dataset, indices, transform, noise_std=0.15):
        self.base_dataset = base_dataset
        self.indices      = list(indices)
        self.transform    = transform
        self.noise_std    = noise_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_pil, _ = self.base_dataset[self.indices[idx]]

        # Apply base transform → clean tensor in [-1, 1]
        clean = self.transform(img_pil)

        # Add Gaussian noise
        noise = torch.randn_like(clean) * self.noise_std
        noisy = torch.clamp(clean + noise, -1.0, 1.0)

        return noisy, clean   # (input, target)


# ============================================================
# MODEL — Convolutional Autoencoder
# ============================================================
class ConvAutoencoder(nn.Module):
    """
    Encoder: 3 × (Conv → BN → ReLU → MaxPool)   128 → 16 spatial
    Decoder: 3 × (ConvTranspose → BN → ReLU)     16 → 128 spatial
    Final layer: Conv + Tanh (output in [-1, 1])
    """
    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────
        self.encoder = nn.Sequential(
            # Block 1: 128×128 → 64×64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64×64 → 32×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32×32 → 16×16  (latent space)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # ── Decoder ──────────────────────────────────────────
        self.decoder = nn.Sequential(
            # Block 1: 16×16 → 32×32
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2: 32×32 → 64×64
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 3: 64×64 → 128×128
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
        )

        # Final activation — keeps output in [-1, 1] matching normalization
        self.out_act = nn.Tanh()

    def forward(self, x):
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return self.out_act(recon)


# ============================================================
# METRICS
# ============================================================
def psnr(clean, recon, max_val=2.0):
    """PSNR between tensors in [-1, 1] (range = 2.0)"""
    mse = torch.mean((clean - recon) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        recon = model(noisy)
        loss  = criterion(recon, clean)
        loss.backward()
        optimizer.step()

        bs = noisy.size(0)
        total_loss += loss.item() * bs
        total_psnr += psnr(clean.detach(), recon.detach()).item() * bs
        n += bs

    return total_loss / n, total_psnr / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            recon = model(noisy)
            loss  = criterion(recon, clean)

            bs = noisy.size(0)
            total_loss += loss.item() * bs
            total_psnr += psnr(clean, recon).item() * bs
            n += bs

    return total_loss / n, total_psnr / n


# ============================================================
# VISUALISATION — before / after grid
# ============================================================
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

    def denorm(t):
        """[-1,1] → [0,1] for display"""
        return (t * 0.5 + 0.5).clamp(0, 1)

    noisy_imgs = torch.cat(noisy_imgs)[:n]
    clean_imgs = torch.cat(clean_imgs)[:n]
    recon_imgs = torch.cat(recon_imgs)[:n]

    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 8))
    titles = ['Noisy Input', 'Denoised Output', 'Clean Reference']

    for i, (row_imgs, title) in enumerate(zip([noisy_imgs, recon_imgs, clean_imgs], titles)):
        axes[i, 0].set_ylabel(title, fontsize=11, fontweight='bold')
        for j in range(n):
            img = denorm(row_imgs[j]).permute(1, 2, 0).numpy()
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

    plt.suptitle('Autoencoder Denoising — MRI Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✓ Visual results saved to '{save_path}'")


# ============================================================
# DENOISE & SAVE — run saved model on a folder of images
# ============================================================
def denoise_folder(model_path, input_folder, output_folder):
    """
    Utility: load trained autoencoder and denoise all images
    in input_folder, saving results to output_folder.
    Use this to preprocess images before passing to BrainTumorCNN.
    """
    os.makedirs(output_folder, exist_ok=True)

    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = Image.open(os.path.join(input_folder, fname)).convert('RGB')
        tensor = base_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            denoised = model(tensor).squeeze(0).cpu()

        # Denormalize → PIL → save
        denoised = (denoised * 0.5 + 0.5).clamp(0, 1)
        denoised_pil = transforms.ToPILImage()(denoised)
        denoised_pil.save(os.path.join(output_folder, fname))

    print(f"✓ Denoised images saved to '{output_folder}'")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("MRI DENOISING - CONVOLUTIONAL AUTOENCODER")
    print("=" * 60)

    # Load dataset (labels are ignored — autoencoder is unsupervised)
    source = datasets.ImageFolder('data/Training', transform=None)
    print(f"\nDataset loaded: {len(source)} images | Classes: {source.classes}")

    # Train / val split
    val_size   = int(VAL_SPLIT * len(source))
    train_size = len(source) - val_size
    train_idx, val_idx = random_split(
        range(len(source)), [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_ds = NoisyMRIDataset(source, train_idx, base_transform, noise_std=NOISE_STD)
    val_ds   = NoisyMRIDataset(source, val_idx,   base_transform, noise_std=NOISE_STD)

    pin = DEVICE.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model     = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {DEVICE} | Params: {total_params:,}")
    print(f"Noise STD: {NOISE_STD} | Loss: MSE | Optimizer: Adam\n")

    # Training loop
    train_losses, val_losses = [], []
    train_psnrs,  val_psnrs  = [], []
    best_val_loss = float('inf')

    print(f"Training for {NUM_EPOCHS} epochs...")
    print("-" * 70)

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_psnr = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_psnr = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_psnrs.append(tr_psnr)
        val_psnrs.append(vl_psnr)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), 'autoencoder.pth')
            flag = '  * saved'
        else:
            flag = ''

        scheduler.step(vl_loss)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f}  PSNR: {tr_psnr:.2f}dB | "
              f"Val Loss: {vl_loss:.4f}  PSNR: {vl_psnr:.2f}dB | "
              f"LR: {lr:.6f}{flag}")

    print("-" * 70)
    print(f"[OK] Training complete | Best Val Loss: {best_val_loss:.4f}")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=4)
    axes[0].plot(val_losses,   label='Val Loss',   marker='s', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Reconstruction Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_psnrs, label='Train PSNR', marker='o', markersize=4)
    axes[1].plot(val_psnrs,   label='Val PSNR',   marker='s', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Peak Signal-to-Noise Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Autoencoder Training | Best Val Loss: {best_val_loss:.4f}')
    plt.tight_layout()
    plt.savefig('autoencoder_training_curves.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("✓ Training curves saved to 'autoencoder_training_curves.png'")

    # Before / after visual
    show_denoising_results(model, val_loader, DEVICE, n=6)

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print("  autoencoder.pth               ← trained model weights")
    print("  autoencoder_training_curves.png")
    print("  denoising_results.png         ← before/after visual")
    print()
    print("NEXT STEP — use denoised images with your classifier:")
    print("  denoise_folder('autoencoder.pth', 'data/Training', 'data/Training_denoised')")
    print("  Then point your BrainTumorCNN training at 'data/Training_denoised'")
    print("=" * 60)


if __name__ == "__main__":
    main()