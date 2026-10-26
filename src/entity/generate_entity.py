from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.entity.dataset_entity import IS_KAGGLE, KAGGLE_WORKING, LOCAL_ROOT
from src.entity.lora_entity import KAGGLE_DATASET


@dataclass
class GenerateEntity:
    """Constants for generating the evaluation images.

    run_name="base" with lora_path=None  -> baseline (untouched Stable Diffusion).
    run_name="rank8" with lora_path=.../lora_final -> fine-tuned model.
    Both use the same prompts, seeds and settings from eval_prompts.json,
    so the two image sets can be compared one to one.
    """

    run_name: str = "base"
    pretrained_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_path: Optional[Path] = None        # folder with pytorch_lora_weights.safetensors
    lora_scale: float = 1.0                 # 0 = no LoRA effect, 1 = full effect
    mixed_precision: bool = True            # fp16 on GPU
    skip_existing: bool = True              # lets an interrupted Kaggle session continue
    seeds: Optional[List[int]] = None       # None -> seeds from eval_prompts.json

    eval_prompts_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if IS_KAGGLE:
            self.eval_prompts_path = Path(self.eval_prompts_path or KAGGLE_DATASET / "eval_prompts.json")
            self.output_dir = Path(self.output_dir or KAGGLE_WORKING / "generated" / self.run_name)
        else:
            self.eval_prompts_path = Path(self.eval_prompts_path or LOCAL_ROOT / "eval_prompts.json")
            self.output_dir = Path(self.output_dir or LOCAL_ROOT / "generated" / self.run_name)
        if self.lora_path is not None:
            self.lora_path = Path(self.lora_path)
