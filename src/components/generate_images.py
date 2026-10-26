import json
from src.logging import logger
from pathlib import Path
from typing import List, Optional

import torch

from src.entity.generate_entity import GenerateEntity



class GenerateImagesComponent:
    """Generates every prompt x seed from eval_prompts.json and writes a manifest.

    Images go to <output_dir>/<category>/prompt<id>_seed<seed>.png
    The manifest (manifest.json) lists prompt, category, seed and file for each image,
    which the evaluation step reads later.

    `pipe` can be passed to inject a ready pipeline (used by the smoke test).
    """

    def __init__(self, config: GenerateEntity, pipe=None) -> None:
        self.cfg = config
        self.pipe = pipe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(config.mixed_precision and self.device.type == "cuda")

    def _build_pipe(self):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        cfg = self.cfg
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.pretrained_model,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # same sampler as the progress samples during training
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if cfg.lora_path is not None:
            if not cfg.lora_path.exists():
                raise FileNotFoundError(f"LoRA folder not found: {cfg.lora_path}")
            pipe.load_lora_weights(str(cfg.lora_path))
            logger.logging.info("Loaded LoRA from %s (scale %.2f)", cfg.lora_path, cfg.lora_scale)
        return pipe.to(self.device)

    def run(self) -> Path:
        cfg = self.cfg
        data = json.loads(cfg.eval_prompts_path.read_text(encoding="utf-8"))
        settings, prompts = data["settings"], data["prompts"]
        seeds = cfg.seeds or settings["seeds"]

        pipe = self.pipe or self._build_pipe()
        pipe.set_progress_bar_config(disable=True)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {}
        if cfg.lora_path is not None and cfg.lora_scale != 1.0:
            kwargs["cross_attention_kwargs"] = {"scale": cfg.lora_scale}

        manifest: List[dict] = []
        total = len(prompts) * len(seeds)
        done = 0
        for p in prompts:
            folder = cfg.output_dir / p["category"]
            folder.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                file = folder / f"prompt{p['id']:02d}_seed{seed}.png"
                if not (cfg.skip_existing and file.exists()):
                    generator = torch.Generator(device=self.device.type).manual_seed(seed)
                    image = pipe(
                        p["prompt"],
                        negative_prompt=settings.get("negative_prompt") or None,
                        num_inference_steps=settings["num_inference_steps"],
                        guidance_scale=settings["guidance_scale"],
                        height=settings["height"], width=settings["width"],
                        generator=generator, **kwargs,
                    ).images[0]
                    image.save(file)
                done += 1
                if done % 10 == 0 or done == total:
                    logger.logging.info("Generated %d/%d", done, total)
                manifest.append({
                    "id": p["id"], "category": p["category"], "group": p.get("group"),
                    "prompt": p["prompt"], "seed": seed, "file": str(file.relative_to(cfg.output_dir)),
                })

        path = cfg.output_dir / "manifest.json"
        path.write_text(json.dumps({
            "run_name": cfg.run_name,
            "lora_path": str(cfg.lora_path) if cfg.lora_path else None,
            "lora_scale": cfg.lora_scale,
            "settings": settings,
            "images": manifest,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.logging.info("Manifest written: %s", path)
        return path
