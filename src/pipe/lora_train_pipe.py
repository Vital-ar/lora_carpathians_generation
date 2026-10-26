from src.logging import logger
from pathlib import Path
from typing import Dict, Optional

from src.components.lora_train import LoraTrainer
from src.entity.lora_entity import LoraTrainEntity

 

def run_lora_train_pipe(config: Optional[LoraTrainEntity] = None, models: Optional[Dict] = None) -> Path:
    """Fine-tune Stable Diffusion with LoRA and return the folder with the final LoRA weights.

    Usage:
        run_lora_train_pipe()                                               # defaults
        run_lora_train_pipe(LoraTrainEntity(rank=16, run_name="rank16"))    # one experiment
    `models` is only for tests (tiny random models, no download).
    """
    config = config or LoraTrainEntity(lora_alpha=16)
    logger.logging.info("LoRA training started: %s", config)
    final = LoraTrainer(config, models=models).run()
    logger.logging.info("LoRA training finished: %s", final)
    return final


if __name__ == "__main__":
    run_lora_train_pipe()
