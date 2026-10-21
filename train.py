
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent

# Paths can be overridden by environment variables on Kaggle.
DATA_ROOT = Path(
    os.environ.get(
        "MOUNTAIN_DATA_DIR",
        str(ROOT / "data" / "sd_lora_dataset"),
    )
).resolve()

OUTPUT_DIR = Path(
    os.environ.get(
        "MOUNTAIN_OUTPUT_DIR",
        str(ROOT / "outputs" / "mountain-lora"),
    )
).resolve()

DIFFUSERS_SCRIPT = (
    ROOT
    / "third_party"
    / "diffusers"
    / "examples"
    / "text_to_image"
    / "train_text_to_image_lora.py"
)

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TRIGGER = "crpth"

# Initial experiment settings.
RESOLUTION = 512
LORA_RANK = 16
BATCH_PER_GPU = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 1000
CHECKPOINT_STEPS = 100
NUM_GPUS = 2
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_split(split_dir: Path) -> set[str]:
    metadata_path = split_dir / "metadata.jsonl"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    records = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{metadata_path}:{line_number}: invalid JSON"
                ) from exc

            filename = record.get("file_name")
            caption = record.get("text")

            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(
                    f"{metadata_path}:{line_number}: invalid file_name"
                )

            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(
                    f"{metadata_path}:{line_number}: empty text caption"
                )

            image_path = (split_dir / filename).resolve()

            if not image_path.is_relative_to(split_dir.resolve()):
                raise ValueError(f"Invalid image path: {filename}")

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {image_path}")

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise ValueError(f"Unreadable image: {image_path}") from exc

            # Use the same unique token in every style-training caption.
            if TRIGGER not in caption:
                raise ValueError(
                    f"{filename}: caption must include trigger {TRIGGER!r}"
                )

            records.append(filename)

    if not records:
        raise ValueError(f"No image records found in {metadata_path}")

    if len(records) != len(set(records)):
        raise ValueError(f"Duplicate image references in {metadata_path}")

    print(f"{split_dir.name}: {len(records)} valid image-caption pairs")
    return set(records)


def validate_dataset() -> None:
    train_files = validate_split(DATA_ROOT / "train")
    holdout_files = validate_split(DATA_ROOT / "holdout")

    overlap = train_files & holdout_files
    if overlap:
        raise ValueError(
            f"Train and holdout share {len(overlap)} filenames. "
            "Keep the splits separate."
        )

    print("Dataset validation passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset without requiring a GPU.",
    )
    args = parser.parse_args()

    validate_dataset()

    if args.validate_only:
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use --validate-only locally, "
            "or enable the GPU accelerator on Kaggle."
        )

    gpu_count = torch.cuda.device_count()
    print(f"Visible GPUs: {gpu_count}")

    if gpu_count != NUM_GPUS:
        raise RuntimeError(
            f"Expected exactly {NUM_GPUS} GPUs, found {gpu_count}. "
            "Check Kaggle accelerator settings."
        )

    if not DIFFUSERS_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Training script missing: {DIFFUSERS_SCRIPT}. "
            "Clone the repository with --recurse-submodules."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = [
        str(DIFFUSERS_SCRIPT),
        "--pretrained_model_name_or_path", BASE_MODEL,
        "--train_data_dir", str(DATA_ROOT / "train"),
        "--image_column", "image",
        "--caption_column", "text",
        "--resolution", str(RESOLUTION),
        "--rank", str(LORA_RANK),
        "--train_batch_size", str(BATCH_PER_GPU),
        "--gradient_accumulation_steps", str(GRAD_ACCUMULATION),
        "--learning_rate", str(LEARNING_RATE),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(MAX_TRAIN_STEPS),
        "--mixed_precision", "fp16",
        "--gradient_checkpointing",
        "--checkpointing_steps", str(CHECKPOINT_STEPS),
        "--checkpoints_total_limit", "2",
        "--dataloader_num_workers", "2",
        "--validation_prompt",
        f"{TRIGGER}, a dramatic mountain landscape in a distinctive visual style",
        "--num_validation_images", "2",
        "--validation_epochs", "5",
        "--seed", str(SEED),
        "--output_dir", str(OUTPUT_DIR),
        "--report_to", "tensorboard",
    ]

    launch_args = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes", str(NUM_GPUS),
        "--mixed_precision", "fp16",
        *training_args,
    ]

    print(f"Dataset: {DATA_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print("Effective batch size:", NUM_GPUS * BATCH_PER_GPU * GRAD_ACCUMULATION)
    print("Starting distributed LoRA training...")

    subprocess.run(launch_args, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
