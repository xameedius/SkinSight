import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

ART = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ART / "skinsight_model.pt"
CLASS_PATH = ART / "class_names.json"
META_PATH = ART / "model_meta.json"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_class_names = None
_tfms = None

def load_model_once():
    global _model, _class_names, _tfms

    if _model is not None:
        return

    with open(CLASS_PATH, "r") as f:
        _class_names = json.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    model_name = meta["model_name"]
    img_size = meta["img_size"]

    _tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    _model = timm.create_model(model_name, pretrained=False, num_classes=len(_class_names))
    state = torch.load(MODEL_PATH, map_location=_device)
    _model.load_state_dict(state)
    _model.eval()
    _model.to(_device)

def predict_pil(pil_img: Image.Image):
    load_model_once()
    x = _tfms(pil_img.convert("RGB")).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

    conf, idx = torch.max(probs, dim=0)
    return {
        "label": _class_names[int(idx)],
        "confidence": float(conf.item()),
        "top3": [
            {"label": _class_names[i], "p": float(probs[i].item())}
            for i in torch.topk(probs, k=min(3, len(_class_names))).indices.tolist()
        ]
    }