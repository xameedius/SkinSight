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

# ---- Paths  ----
ART_DIR = Path("artifacts")
DEFAULT_EVAL_DIR = Path("data/test")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eval_transforms(img_size: int):
    # Must match your val transforms normalization
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def load_model(device: str):
    meta_path = ART_DIR / "model_meta.json"
    classes_path = ART_DIR / "class_names.json"
    weights_path = ART_DIR / "skinsight_model.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing {classes_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}")

    meta = load_json(meta_path)
    class_names = load_json(classes_path)

    model_name = meta.get("model_name", "tf_efficientnetv2_s")
    img_size = int(meta.get("img_size", 224))
    num_classes = len(class_names)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    return model, class_names, img_size, model_name


def plot_confusion_matrix(cm: np.ndarray, labels, out_path: Path, normalize=False):
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""),
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, img_size, model_name = load_model(device)

    eval_dir = DEFAULT_EVAL_DIR
    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Evaluation directory not found: {eval_dir}\n"
            f"Expected: /content/data/test/<class_name>/*.jpg"
        )

    tfm = build_eval_transforms(img_size)
    ds = ImageFolder(eval_dir, transform=tfm)

    # Check class order mismatch (IMPORTANT)
    if ds.classes != class_names:
        print("⚠️ CLASS ORDER MISMATCH")
        print("ImageFolder classes:", ds.classes[:5], " ...")
        print("class_names.json   :", class_names[:5], " ...")
        print("\nThis can break evaluation and your Django label mapping.\n"
              "Fix options:\n"
              "1) Ensure folder names match exactly and order matches training\n"
              "2) Rebuild test folder to match class_names.json\n"
              "3) Retrain and regenerate class_names.json from the same folder structure\n")

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

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

    ART_DIR.mkdir(exist_ok=True)

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

    with open(ART_DIR / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, ds.classes, ART_DIR / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, ds.classes, ART_DIR / "confusion_matrix_norm.png", normalize=True)

    with open(ART_DIR / "eval_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "true_label", "pred_label", "confidence"])
        for (path, true_idx), pred_idx, conf in zip(ds.samples, y_pred, y_conf):
            w.writerow([path, ds.classes[true_idx], ds.classes[pred_idx], f"{conf:.6f}"])

    print("\n✅ Evaluation complete")
    print("Model:", model_name)
    print("Eval dir:", eval_dir)
    print("Samples:", len(ds))
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision/Recall/F1: {prec_macro:.4f} / {rec_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted Precision/Recall/F1: {prec_weight:.4f} / {rec_weight:.4f} / {f1_weight:.4f}")
    print("\nClassification report:\n")
    print(report)
    print("\nSaved to /content/artifacts/")
    print("- eval_metrics.json")
    print("- confusion_matrix.png")
    print("- confusion_matrix_norm.png")
    print("- eval_predictions.csv")


if __name__ == "__main__":
    main()