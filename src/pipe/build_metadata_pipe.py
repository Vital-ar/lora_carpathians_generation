from src.components.build_metadata import BuildMetadataComponent
from src.entity.dataset_entity import DatasetPrepEntity

from src.logging import logger
import logging


def run_dataset_prep_pipe():
    """Build train/holdout folders with metadata.jsonl.

    Usage:
        run_dataset_prep_pipe()                                  # defaults from the entity
        run_dataset_prep_pipe(DatasetPrepEntity(holdout=6))      # override some constants
    """
    config = DatasetPrepEntity()
    logger.logging.info("Dataset preparation started: %s", config)
    result = BuildMetadataComponent(config).run()
    logger.logging.info("Dataset preparation finished: %s", {k: str(v) for k, v in result.items()})
    return result


if __name__ == "__main__":

    run_dataset_prep_pipe()