from src.logging import logger
from pathlib import Path
from typing import Optional

from src.components.generate_images import GenerateImagesComponent
from src.entity.generate_entity import GenerateEntity



def run_generate_images_pipe(config: Optional[GenerateEntity] = None, pipe=None) -> Path:
    """Generate all evaluation images and return the path to manifest.json.

    Usage:
        run_generate_images_pipe()                                       # baseline (no LoRA)
        run_generate_images_pipe(GenerateEntity(run_name="rank8",
                                                lora_path="runs/baseline-rank8/lora_final"))
    `pipe` is only for tests.
    """
    config = config or GenerateEntity()
    logger.logging.info("Image generation started: %s", config)
    manifest = GenerateImagesComponent(config, pipe=pipe).run()
    logger.logging.info("Image generation finished: %s", manifest)
    return manifest


if __name__ == "__main__":
    run_generate_images_pipe()
