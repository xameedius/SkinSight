import os, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import timm
from timm.data.mixup import Mixup
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.utils import ModelEmaV2
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


# -----------------------
# Config
# -----------------------
DATA_ROOT = Path("/content/data")
ART_DIR = Path("artifacts")

MODEL_NAME = "tf_efficientnetv2_s"
IMG_SIZE = 320
BATCH_SIZE = 8
EPOCHS = 28
PATIENCE = 5
WARMUP_EPOCHS = 1

LR_HEAD = 7e-4
LR_BACKBONE = 1.2e-4
MIN_LR = 8e-6
WEIGHT_DECAY = 8e-5

NUM_WORKERS = 2
SEED = 42
USE_COMPILE = False

MIXUP_ALPHA = 0.03
CUTMIX_ALPHA = 0.0
MIXUP_PROB = 0.20
LABEL_SMOOTHING = 0.03

EMA_DECAY = 0.9997
GRAD_CLIP = 1.0


# -----------------------
# Utils
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_transforms(img_size: int):
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.03),
        transforms.RandomRotation(degrees=6),
        transforms.ColorJitter(
            brightness=0.06,
            contrast=0.06,
            saturation=0.05,
            hue=0.01
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])
    return train_tfm, val_tfm


class WeightedSoftTargetCrossEntropy(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)

        if self.class_weights is not None:
            weights = self.class_weights.unsqueeze(0)
            loss = -(target * log_probs * weights).sum(dim=-1)
            norm = (target * weights).sum(dim=-1).clamp_min(1e-8)
            loss = loss / norm
        else:
            loss = -(target * log_probs).sum(dim=-1)

        return loss.mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / max(total, 1)


def set_trainable_layers(model, head_only=True):
    for name, p in model.named_parameters():
        if head_only:
            if any(k in name.lower() for k in ["classifier", "head", "fc"]):
                p.requires_grad = True
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True


def build_optimizer(model, head_only=True):
    if head_only:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    head_params = []
    backbone_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name.lower() for k in ["classifier", "head", "fc"]):
            head_params.append(p)
        else:
            backbone_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


# -----------------------
# Training
# -----------------------
def main():
    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if device == "cuda":
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"

    train_tfm, val_tfm = build_transforms(IMG_SIZE)
    train_ds = ImageFolder(train_dir, transform=train_tfm)
    val_ds = ImageFolder(val_dir, transform=val_tfm)

    class_names = train_ds.classes
    num_classes = len(class_names)

    ART_DIR.mkdir(exist_ok=True)
    save_json(ART_DIR / "class_names.json", class_names)

    y_train = np.array([y for _, y in train_ds.samples])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.15,
        drop_path_rate=0.08,
    )

    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    if device == "cuda" and USE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("✅ torch.compile enabled")
        except Exception as e:
            print(f"⚠️ torch.compile skipped: {e}")

    set_trainable_layers(model, head_only=True)
    optimizer = build_optimizer(model, head_only=True)

    steps_per_epoch = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=EPOCHS * steps_per_epoch,
        lr_min=MIN_LR,
        warmup_t=max(1, steps_per_epoch),
        warmup_lr_init=MIN_LR,
        cycle_limit=1,
        t_in_epochs=False,
    )

    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=CUTMIX_ALPHA,
        prob=MIXUP_PROB,
        switch_prob=0.0,
        mode="batch",
        label_smoothing=LABEL_SMOOTHING,
        num_classes=num_classes,
    ) if MIXUP_ALPHA > 0 else None

    criterion_soft = WeightedSoftTargetCrossEntropy(class_weights=class_weights)
    criterion_hard = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    model_ema = ModelEmaV2(model, decay=EMA_DECAY, device=device)

    best_val_acc = -1.0
    bad_epochs = 0
    global_step = 0

    save_json(ART_DIR / "model_meta.json", {
        "model_name": MODEL_NAME,
        "img_size": IMG_SIZE,
        "num_classes": num_classes,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "class_weights": class_weights.detach().cpu().tolist(),
        "mixup_alpha": MIXUP_ALPHA,
        "mixup_prob": MIXUP_PROB,
        "seed": SEED,
    })

    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            print("🔥 Unfreezing full model")
            set_trainable_layers(model, head_only=False)
            optimizer = build_optimizer(model, head_only=False)

            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=(EPOCHS - epoch + 1) * steps_per_epoch,
                lr_min=MIN_LR,
                warmup_t=max(1, steps_per_epoch // 2),
                warmup_lr_init=MIN_LR,
                cycle_limit=1,
                t_in_epochs=False,
            )
            global_step = 0

        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                if mixup_fn is not None:
                    x_mix, y_soft = mixup_fn(x, y)
                    logits = model(x_mix)
                    loss = criterion_soft(logits, y_soft)
                else:
                    logits = model(x)
                    loss = criterion_hard(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            model_ema.update(model)

            scheduler.step_update(global_step)
            global_step += 1

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_ds)
        val_acc = evaluate(model_ema.module, val_loader, device)

        print(f"\nEpoch {epoch} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0

            tmp_ckpt = ART_DIR / "skinsight_model.tmp"
            final_ckpt = ART_DIR / "skinsight_model.pt"
            torch.save(model_ema.module.state_dict(), tmp_ckpt)
            tmp_ckpt.replace(final_ckpt)

            save_json(ART_DIR / "best_metrics.json", {
                "best_val_acc": float(best_val_acc),
                "epoch": epoch,
            })
            print("✅ Saved best EMA model")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("🛑 Early stopping")
                break

    print("\nTraining complete")
    print("Best val accuracy:", best_val_acc)


if __name__ == "__main__":
    main()