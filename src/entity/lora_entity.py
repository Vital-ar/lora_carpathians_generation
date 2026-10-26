from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.entity.dataset_entity import IS_KAGGLE, KAGGLE_WORKING, LOCAL_ROOT

# Change to the slug of the Kaggle dataset that holds sd_lora_dataset/ and eval_prompts.json
KAGGLE_DATASET = Path("data")


@dataclass
class LoraTrainEntity:
    """All constants for LoRA fine-tuning of Stable Diffusion 1.5.

    Override any field when creating the object, e.g.
    LoraTrainEntity(rank=16, learning_rate=5e-5, run_name="rank16").
    """

    # ---- model ----
    pretrained_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    resolution: int = 512

    # ---- LoRA ----
    rank: int = 8
    lora_alpha: Optional[int] = None  # None -> equals rank
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])

    # ---- optimisation ----
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lr_scheduler: str = "cosine"          # "constant", "cosine", "linear", ...
    lr_warmup_steps: int = 100
    max_train_steps: int = 1000             # optimizer steps (after accumulation)
    batch_size: int = 1
    grad_accum_steps: int = 8             # effective batch = batch_size * grad_accum_steps
    max_grad_norm: float = 1.0
    mixed_precision: bool = True            # fp16 autocast on GPU (T4 has no bf16)
    gradient_checkpointing: bool = True
    horizontal_flip_prob: float = 0.5       # fine for landscapes
    num_workers: int = 2
    seed: int = 42

    # ---- checkpoints, samples, logging ----
    checkpoint_every: int = 100
    sample_every: int = 100
    num_sample_prompts: int = 4             # prompts from eval_prompts.json used for progress samples
    resume: bool = True
    use_mlflow: bool = True
    mlflow_experiment: str = "sd-lora-carpathians"
    run_name: str = "baseline-rank8"

    # ---- paths (None -> chosen automatically) ----
    train_dir: Optional[Path] = None
    eval_prompts_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.lora_alpha is None:
            self.lora_alpha = self.rank
        if IS_KAGGLE:
            self.train_dir = Path(self.train_dir or KAGGLE_DATASET / "sd_lora_dataset" / "train")
            self.eval_prompts_path = Path(self.eval_prompts_path or KAGGLE_DATASET / "eval_prompts.json")
            self.output_dir = Path(self.output_dir or KAGGLE_WORKING / "runs" / self.run_name)
        else:
            self.train_dir = Path(self.train_dir or LOCAL_ROOT / "sd_lora_dataset" / "train")
            self.eval_prompts_path = Path(self.eval_prompts_path or LOCAL_ROOT / "eval_prompts.json")
            self.output_dir = Path(self.output_dir or LOCAL_ROOT / "runs" / self.run_name)
