import os, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


# -----------------------
# Config (HIGH ACCURACY RUN)
# -----------------------
DATA_ROOT = Path("/content/data")
ART_DIR = Path("artifacts")

MODEL_NAME = "tf_efficientnetv2_s"

IMG_SIZE = 320          # BIGGEST improvement for derm classification
BATCH_SIZE = 8          # safe for T4 at 320px
EPOCHS = 25
PATIENCE = 7
WARMUP_EPOCHS = 2       # freeze backbone first

LR = 3e-4
WEIGHT_DECAY = 2e-4

NUM_WORKERS = 2
SEED = 42

# Reduced mixing (preserve medical detail)
MIXUP_ALPHA = 0.05
CUTMIX_ALPHA = 0.3
LABEL_SMOOTHING = 0.1


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
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.1),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    return train_tfm, val_tfm


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return correct / total


# -----------------------
# Training
# -----------------------
def main():
    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dir = DATA_ROOT / "train"
    val_dir   = DATA_ROOT / "val"

    train_tfm, val_tfm = build_transforms(IMG_SIZE)
    train_ds = ImageFolder(train_dir, transform=train_tfm)
    val_ds   = ImageFolder(val_dir, transform=val_tfm)

    class_names = train_ds.classes
    num_classes = len(class_names)

    ART_DIR.mkdir(exist_ok=True)
    save_json(ART_DIR / "class_names.json", class_names)

    # class weights
    y_train = np.array([y for _, y in train_ds.samples])
    cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    cw = torch.tensor(cw, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model.to(device)

    # ---- FREEZE BACKBONE FOR WARMUP ----
    for name, p in model.named_parameters():
        if "classifier" in name or "head" in name or "fc" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    use_mix = (MIXUP_ALPHA > 0) or (CUTMIX_ALPHA > 0)
    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=CUTMIX_ALPHA,
        prob=1.0,
        switch_prob=0.5,
        label_smoothing=LABEL_SMOOTHING,
        num_classes=num_classes,
    ) if use_mix else None

    criterion = SoftTargetCrossEntropy() if mixup_fn else nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=EPOCHS * steps_per_epoch,
        lr_min=LR * 0.05,
        warmup_t=max(1, int(0.5 * steps_per_epoch)),
        warmup_lr_init=LR * 0.1,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device=="cuda"))

    best_val_acc = -1
    bad_epochs = 0
    global_step = 0

    save_json(ART_DIR / "model_meta.json", {
        "model_name": MODEL_NAME,
        "img_size": IMG_SIZE,
        "num_classes": num_classes
    })

    for epoch in range(1, EPOCHS+1):

        # ---- UNFREEZE AFTER WARMUP ----
        if epoch == WARMUP_EPOCHS + 1:
            print("🔥 Unfreezing full model")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        model.train()
        running_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            if mixup_fn:
                x, y_mix = mixup_fn(x, y)
            else:
                y_mix = y

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device=="cuda")):
                logits = model(x)
                loss = criterion(logits, y_mix)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step_update(global_step)
            global_step += 1

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), ART_DIR / "skinsight_model.pt")
            print("✅ Saved best model")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("🛑 Early stopping")
                break

    print("\nTraining complete")
    print("Best val accuracy:", best_val_acc)


if __name__ == "__main__":
    main()