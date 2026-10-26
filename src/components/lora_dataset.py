import json
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CaptionImageDataset(Dataset):
    """Reads <folder>/metadata.jsonl ({"file_name": ..., "text": ...} per line).

    Returns pixel_values in [-1, 1] and the tokenized caption (padded to the
    tokenizer's max length), as Stable Diffusion expects."""

    def __init__(self, folder: Path, tokenizer, resolution: int = 512, flip_prob: float = 0.5) -> None:
        self.folder = Path(folder)
        self.tokenizer = tokenizer
        meta = self.folder / "metadata.jsonl"
        if not meta.exists():
            raise FileNotFoundError(f"{meta} not found")
        self.items = [json.loads(l) for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self.items:
            raise ValueError(f"{meta} is empty")
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),  # no-op for images already resized to resolution
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        with Image.open(self.folder / item["file_name"]) as img:
            pixel_values = self.transform(img.convert("RGB"))
        input_ids = self.tokenizer(
            item["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "input_ids": input_ids}
