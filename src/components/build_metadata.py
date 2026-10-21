import csv
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from src.entity.dataset_entity import DatasetPrepEntity

from src.logging import logger
Pair = Tuple[str, str]


class BuildMetadataComponent:
    """Pairs images with captions, validates them, splits train/holdout and
    writes one metadata.jsonl per split (Hugging Face imagefolder format)."""

    def __init__(self, config: DatasetPrepEntity) -> None:
        self.cfg = config

    # ---------- reading ----------
    def _list_images(self) -> List[str]:
        return sorted(
            p.name for p in self.cfg.images_dir.iterdir()
            if p.suffix.lower() in self.cfg.image_exts
        )

    def _read_captions(self) -> Dict[str, str]:
        path = self.cfg.captions_path
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r["file_name"].strip()] = r["text"].strip()
            return rows
        if suffix == ".csv":
            with open(path, newline="", encoding="utf-8") as f:
                return {r["image_name"].strip(): r["caption"].strip() for r in csv.DictReader(f)}
        raise ValueError(f"Unsupported captions file type: {suffix} (use .jsonl or .csv)")

    def read_pairs(self) -> List[Pair]:
        images = self._list_images()
        captions = self._read_captions()
        missing = [n for n in images if n not in captions]
        extra = [n for n in captions if n not in images]
        if missing or extra:
            raise ValueError(f"Images without caption: {missing}\nCaptions without image: {extra}")
        return [(n, captions[n]) for n in images]

    # ---------- validation ----------
    def validate_and_fix(self, pairs: List[Pair]) -> List[Pair]:
        cfg, cleaned = self.cfg, []
        for name, text in pairs:
            with Image.open(cfg.images_dir / name) as im:
                im.verify()
            with Image.open(cfg.images_dir / name) as im:
                if min(im.size) < cfg.min_side:
                    logger.logging.warning("%s: small image %s", name, im.size)
            if cfg.trigger and not text.startswith(cfg.trigger):
                if cfg.replace_prefix and text.startswith(cfg.replace_prefix):
                    text = cfg.trigger + text[len(cfg.replace_prefix):]
                else:
                    text = f"{cfg.trigger}, {text}"
                logger.logging.info("%s: trigger set", name)
            if len(text.split()) > cfg.max_words:
                logger.logging.warning("%s: caption is long (%d words)", name, len(text.split()))
            cleaned.append((name, text))
        return cleaned

    # ---------- splitting and writing ----------
    def split(self, pairs: List[Pair]) -> Dict[str, List[Pair]]:
        rng = random.Random(self.cfg.seed)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        hold = set(idx[: self.cfg.holdout])
        splits: Dict[str, List[Pair]] = {"train": [], "holdout": []}
        for i, pair in enumerate(pairs):
            splits["holdout" if i in hold else "train"].append(pair)
        return splits

    def write(self, splits: Dict[str, List[Pair]]) -> Dict[str, Path]:
        written = {}
        for split, items in splits.items():
            folder = self.cfg.out_dir / split
            folder.mkdir(parents=True, exist_ok=True)
            meta = folder / "metadata.jsonl"
            with open(meta, "w", encoding="utf-8") as f:
                for name, text in items:
                    shutil.copy2(self.cfg.images_dir / name, folder / name)
                    f.write(json.dumps({"file_name": name, "text": text}, ensure_ascii=False) + "\n")
            logger.logging.info("%s: %d images -> %s", split, len(items), meta)
            written[split] = meta
        return written

    # ---------- entry point ----------
    def run(self) -> Dict[str, Path]:
        pairs = self.validate_and_fix(self.read_pairs())
        return self.write(self.split(pairs))