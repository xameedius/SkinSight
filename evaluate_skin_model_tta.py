import json, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

ART_DIR = Path("/content/artifacts")
EVAL_DIR = Path("/content/data/test")

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def build_base_tfm(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

def build_flip_tfm(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

def load_model(device):
    meta = load_json(ART_DIR/"model_meta.json")
    class_names = load_json(ART_DIR/"class_names.json")
    model_name = meta.get("model_name", "tf_efficientnetv2_s")
    img_size = int(meta.get("img_size", 224))
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    sd = torch.load(ART_DIR/"skinsight_model.pt", map_location=device)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, class_names, img_size, model_name

@torch.no_grad()
def predict_probs(model, loader, device):
    probs_all = []
    y_all = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        probs_all.append(probs.cpu())
        y_all.extend(y.numpy().tolist())
    return torch.cat(probs_all, dim=0).numpy(), y_all

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, img_size, model_name = load_model(device)

    ds_base = ImageFolder(EVAL_DIR, transform=build_base_tfm(img_size))
    ds_flip = ImageFolder(EVAL_DIR, transform=build_flip_tfm(img_size))

    if ds_base.samples != ds_flip.samples:
        raise RuntimeError("Sample order mismatch between datasets (should never happen).")

    loader_base = DataLoader(ds_base, batch_size=64, shuffle=False, num_workers=2)
    loader_flip = DataLoader(ds_flip, batch_size=64, shuffle=False, num_workers=2)

    probs1, y_true = predict_probs(model, loader_base, device)
    probs2, _ = predict_probs(model, loader_flip, device)

    # TTA: average probabilities
    probs_tta = (probs1 + probs2) / 2.0

    y_pred_single = probs1.argmax(axis=1).tolist()
    y_pred_tta = probs_tta.argmax(axis=1).tolist()
    y_conf_tta = probs_tta.max(axis=1).tolist()

    # Metrics
    acc_single = accuracy_score(y_true, y_pred_single)
    acc_tta = accuracy_score(y_true, y_pred_tta)

    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred_tta, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred_tta, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred_tta, target_names=ds_base.classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_tta)

    out = {
        "model_name": model_name,
        "img_size": img_size,
        "eval_dir": str(EVAL_DIR),
        "num_samples": len(ds_base),
        "accuracy_single": float(acc_single),
        "accuracy_tta": float(acc_tta),
        "precision_macro_tta": float(pm),
        "recall_macro_tta": float(rm),
        "f1_macro_tta": float(fm),
        "precision_weighted_tta": float(pw),
        "recall_weighted_tta": float(rw),
        "f1_weighted_tta": float(fw),
        "classes": ds_base.classes,
        "classification_report_text_tta": report,
        "confusion_matrix_tta": cm.tolist(),
    }

    ART_DIR.mkdir(exist_ok=True)
    with open(ART_DIR/"eval_metrics_tta.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with open(ART_DIR / "eval_predictions_tta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "true_label", "pred_label", "confidence_tta"])
        for (path, true_idx), pred_idx, conf in zip(ds_base.samples, y_pred_tta, y_conf_tta):
            w.writerow([path, ds_base.classes[true_idx], ds_base.classes[pred_idx], f"{conf:.6f}"])

    print("\n✅ TTA Evaluation complete")
    print("Model:", model_name)
    print("Samples:", len(ds_base))
    print(f"Accuracy (single): {acc_single:.4f}")
    print(f"Accuracy (TTA avg flip): {acc_tta:.4f}")
    print(f"Macro P/R/F1 (TTA): {pm:.4f} / {rm:.4f} / {fm:.4f}")
    print(f"Weighted P/R/F1 (TTA): {pw:.4f} / {rw:.4f} / {fw:.4f}")
    print("\nClassification report (TTA):\n")
    print(report)

if __name__ == "__main__":
    main()