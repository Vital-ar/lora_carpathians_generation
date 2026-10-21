from pathlib import Path

from src.components.resize_images import ResizeImagesComponent
from src.entity.dataset_entity import ImageResizeEntity
from src.logging import logger


def run_resize_images_pipe() :

    config =  ImageResizeEntity()
    logger.logging.info("Image resizing started: %s", config)
    saved = ResizeImagesComponent(config).run()
    logger.logging.info("Image resizing finished: %d files", len(saved))
    return saved


if __name__ == "__main__":
    run_resize_images_pipe()