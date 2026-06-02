from __future__ import annotations

from typing import Optional
import torch
from PIL import Image

_model = None
_processor = None

DAMAGE_LABELS = [
    "a car in perfect condition with no damage",
    "a car with minor scratches or small dents",
    "a car with moderate damage and visible dents",
    "a car with severe damage from major accidents",
]

CONDITION_WEIGHTS = torch.tensor([1.0, 0.7, 0.4, 0.1])


def _get_clip():
    global _model, _processor
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()
    return _model, _processor


def get_condition_score(image: Image.Image) -> tuple[float, list[float]]:
    model, processor = _get_clip()

    inputs = processor(
        text=DAMAGE_LABELS,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]
    condition_score = float((probs * CONDITION_WEIGHTS).sum())

    return condition_score, probs.tolist()


def get_condition_label(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    elif score >= 0.65:
        return "Good"
    elif score >= 0.40:
        return "Fair"
    else:
        return "Poor"


def get_condition_display(image: Optional[Image.Image]) -> tuple[float, str, list[float]]:
    if image is None:
        return 0.75, "Good (default — no image provided)", [0.0, 0.75, 0.2, 0.05]
    score, probs = get_condition_score(image)
    label = get_condition_label(score)
    return score, label, probs
