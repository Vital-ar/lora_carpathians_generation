import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# Change these two names to match your own folders / Kaggle dataset slug.
LOCAL_ROOT = Path("data")
KAGGLE_INPUT = Path("/kaggle/input/carpathian-raw")
KAGGLE_WORKING = Path("/kaggle/working/sd_lora")


@dataclass
class DatasetPrepEntity:
    """All constants for building the train/holdout folders with metadata.jsonl.

    Any field can be overridden when creating the object, e.g.
    DatasetPrepEntity(holdout=6, trigger="crpth").
    Paths left as None are chosen automatically for local or Kaggle.
    """

    images_dir: Optional[Path] = None      # folder with the raw images
    captions_path: Optional[Path] = None   # .jsonl, .csv or .txt with captions
    out_dir: Optional[Path] = None         # where train/ and holdout/ are written

    trigger: str = "crpth"                              # rare token for every caption
    replace_prefix: str = "ukrainian carpathian mountains"  # old prefix swapped for the trigger
    holdout: int = 5                       # images kept out of training
    min_side: int = 512                    # warn if the shorter image side is smaller
    max_words: int = 40                    # warn on longer captions
    seed: int = 42                         # fixed seed -> the same holdout every run
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    def __post_init__(self) -> None:
        if IS_KAGGLE:
            self.images_dir = Path(self.images_dir or KAGGLE_INPUT / "images_512")
            self.captions_path = Path(self.captions_path or KAGGLE_INPUT / "excel_metadata.csv")
            self.out_dir = Path(self.out_dir or KAGGLE_WORKING)
        else:
            self.images_dir = Path(self.images_dir or LOCAL_ROOT / "images_512")
            self.captions_path = Path(self.captions_path or LOCAL_ROOT / "excel_metadata.csv")
            self.out_dir = Path(self.out_dir or LOCAL_ROOT / "sd_lora_dataset")




 
@dataclass
class ImageResizeEntity:
    """Constants for rescaling (shorter side -> size) and center-cropping images."""
 
    input_dir: Optional[Path] = None   # folder with the original photos
    output_dir: Optional[Path] = None  # folder for the resized images
    size: int = 512                    # final image is size x size
    jpeg_quality: int = 95
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
 
    def __post_init__(self) -> None:
        # Resizing is light CPU work: do it on the laptop, then upload output_dir to Kaggle.
        self.input_dir = Path(self.input_dir or LOCAL_ROOT / "images")
        self.output_dir = Path(self.output_dir or LOCAL_ROOT / "images_512")
 