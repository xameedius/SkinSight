# evaluate_skin_model.py
import json
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import timm
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

ART = Path("artifacts")
DEFAULT_EVAL_DIR = Path("data") / "test"   # <- set to data/val if that's what you have


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def val_transforms(img_size: int):
    # must match val_tfms in training
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def build_model(device: str):
    meta = load_json(ART / "model_meta.json")
    class_names = load_json(ART / "class_names.json")

    model_name = meta["model_name"]
    img_size = int(meta["img_size"])
    num_classes = len(class_names)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(ART / "skinsight_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names, img_size, model_name


def plot_cm(cm: np.ndarray, labels, out_path: Path, normalize: bool = False):
    mat = cm.astype(np.float64)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ART.mkdir(exist_ok=True)

    # Sanity checks
    for f in ["skinsight_model.pt", "class_names.json", "model_meta.json"]:
        if not (ART / f).exists():
            raise FileNotFoundError(f"Missing artifacts/{f}")

    model, trained_class_names, img_size, model_name = build_model(device)

    eval_dir = DEFAULT_EVAL_DIR
    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Evaluation directory not found: {eval_dir}\n"
            f"Create it like: data/test/<class_name>/*.jpg"
        )

    ds = ImageFolder(eval_dir, transform=val_transforms(img_size))

    # ⚠️ IMPORTANT: class order must match training order
    if ds.classes != trained_class_names:
        print("\n⚠️ WARNING: CLASS ORDER MISMATCH")
        print("Train classes (class_names.json):", trained_class_names)
        print("Eval classes  (ImageFolder):     ", ds.classes)
        print("\nMetrics/confusion matrix will be WRONG unless these match exactly.\n")

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    y_true, y_pred, y_conf = [], [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        conf = torch.max(probs, dim=1).values

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_conf.extend(conf.cpu().numpy().tolist())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_true, y_pred, target_names=ds.classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save outputs
    metrics = {
        "model_name": model_name,
        "img_size": img_size,
        "eval_dir": str(eval_dir),
        "num_samples": len(ds),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weight),
        "recall_weighted": float(rec_weight),
        "f1_weighted": float(f1_weight),
        "classes": ds.classes,
        "classification_report_text": report,
    }

    with open(ART / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_cm(cm, ds.classes, ART / "confusion_matrix.png", normalize=False)
    plot_cm(cm, ds.classes, ART / "confusion_matrix_norm.png", normalize=True)

    with open(ART / "eval_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "true_label", "pred_label", "confidence"])
        for (path, true_idx), pred_idx, conf in zip(ds.samples, y_pred, y_conf):
            w.writerow([path, ds.classes[true_idx], ds.classes[pred_idx], f"{conf:.6f}"])

    print("\n✅ Evaluation complete")
    print("Model:", model_name)
    print("Device:", device)
    print("Eval dir:", eval_dir)
    print("Samples:", len(ds))
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision/Recall/F1: {prec_macro:.4f} / {rec_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted Precision/Recall/F1: {prec_weight:.4f} / {rec_weight:.4f} / {f1_weight:.4f}")
    print("\nClassification report:\n")
    print(report)
    print("\nSaved to artifacts/:")
    print("- eval_metrics.json")
    print("- confusion_matrix.png")
    print("- confusion_matrix_norm.png")
    print("- eval_predictions.csv")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()