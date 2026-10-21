from pathlib import Path
from typing import List

from PIL import Image, ImageOps

from src.entity.dataset_entity import ImageResizeEntity

from src.logging import logger

class ResizeImagesComponent:
    """Rescales the shorter side of every image to `size`, center-crops to
    size x size and saves the result under the SAME file name in output_dir
    (so captions in metadata.jsonl still match)."""

    def __init__(self, config: ImageResizeEntity) -> None:
        self.cfg = config

    def _process_one(self, src: Path, dst: Path) -> None:
        cfg = self.cfg
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)  # apply the phone's rotation flag first
            img = img.convert("RGB")
            if min(img.size) < cfg.size:
                logger.logging.warning("%s: shorter side %d < %d, image is upscaled", src.name, min(img.size), cfg.size)
            # fit() = rescale so the shorter side matches, then center crop
            img = ImageOps.fit(img, (cfg.size, cfg.size), method=Image.LANCZOS, centering=(0.5, 0.5))
            if dst.suffix.lower() in (".jpg", ".jpeg"):
                img.save(dst, quality=cfg.jpeg_quality, subsampling=0)  # EXIF/GPS is not copied
            else:
                img.save(dst)

    def run(self) -> List[Path]:
        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in cfg.input_dir.iterdir() if p.suffix.lower() in cfg.image_exts)
        if not files:
            raise FileNotFoundError(f"No images found in {cfg.input_dir}")
        saved = []
        for src in files:
            dst = cfg.output_dir / src.name
            self._process_one(src, dst)
            saved.append(dst)
        logger.logging.info("Resized %d images to %dx%d -> %s", len(saved), cfg.size, cfg.size, cfg.output_dir)
        return saved