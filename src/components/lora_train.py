import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.components.lora_dataset import CaptionImageDataset
from src.entity.lora_entity import LoraTrainEntity

from src.logging import logger

class LoraTrainer:
    """LoRA fine-tuning of the Stable Diffusion UNet (single GPU or CPU).

    What is written here (and not hidden in a script): data loading, the
    diffusion training step, mixed precision, gradient accumulation,
    resumable checkpoints, progress samples and MLflow logging.
    What comes from libraries: the pretrained model (diffusers) and the
    LoRA layers (peft).

    `models` can be passed to inject small test models (see tests/smoke_test_lora.py);
    keys: tokenizer, text_encoder, vae, unet, noise_scheduler.
    """

    def __init__(self, config: LoraTrainEntity, models: Optional[Dict] = None) -> None:
        self.cfg = config
        self.models = models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(config.mixed_precision and self.device.type == "cuda")
        self.weight_dtype = torch.float16 if self.use_amp else torch.float32

    # ------------------------------------------------------------------ setup
    def _load_models(self) -> None:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        name = self.cfg.pretrained_model
        self.tokenizer = CLIPTokenizer.from_pretrained(name, subfolder="tokenizer")
        self.noise_scheduler = DDPMScheduler.from_pretrained(name, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(name, subfolder="unet")

    def _prepare_models(self) -> None:
        from peft import LoraConfig

        cfg = self.cfg
        if self.models is None:
            self._load_models()
        else:
            self.tokenizer = self.models["tokenizer"]
            self.text_encoder = self.models["text_encoder"]
            self.vae = self.models["vae"]
            self.unet = self.models["unet"]
            self.noise_scheduler = self.models["noise_scheduler"]

        # freeze everything, then add trainable LoRA adapters to the UNet attention layers
        for m in (self.vae, self.text_encoder, self.unet):
            m.requires_grad_(False)
        self.vae.to(self.device, dtype=self.weight_dtype).eval()
        self.text_encoder.to(self.device, dtype=self.weight_dtype).eval()
        self.unet.to(self.device, dtype=self.weight_dtype)

        self.unet.add_adapter(LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=cfg.target_modules,
        ))
        for p in self.unet.parameters():  # trainable LoRA weights stay in fp32 (stable fp16 training)
            if p.requires_grad:
                p.data = p.data.float()
        if cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.params: List[torch.nn.Parameter] = [p for p in self.unet.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in self.params)
        n_total = sum(p.numel() for p in self.unet.parameters())
        logger.logging.info("Trainable LoRA parameters: %d (%.3f%% of the UNet)", n_train, 100 * n_train / n_total)

    def _prepare_data(self) -> None:
        cfg = self.cfg
        ds = CaptionImageDataset(cfg.train_dir, self.tokenizer, cfg.resolution, cfg.horizontal_flip_prob)
        self.loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            drop_last=len(ds) >= cfg.batch_size, pin_memory=self.device.type == "cuda",
        )
        logger.logging.info("Training images: %d, effective batch size: %d", len(ds), cfg.batch_size * cfg.grad_accum_steps)

    def _batches(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:  # reshuffles every pass over the data
            for batch in self.loader:
                yield batch

    # --------------------------------------------------------------- training
    def _loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.weight_dtype)
        input_ids = batch["input_ids"].to(self.device)
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
            text_embeds = self.text_encoder(input_ids)[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {prediction_type}")

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            pred = self.unet(noisy_latents, timesteps, text_embeds).sample
        return F.mse_loss(pred.float(), target.float())

    # ----------------------------------------------------- checkpoints / resume
    def _save_state(self, step: int) -> None:
        from peft import get_peft_model_state_dict

        out = self.cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "state.pt.tmp"
        torch.save({
            "step": step,
            "lora": get_peft_model_state_dict(self.unet),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, tmp)
        os.replace(tmp, out / "state.pt")  # atomic: a crash never leaves a half-written file
        logger.logging.info("State saved at step %d", step)

    def _load_state(self) -> int:
        from peft import set_peft_model_state_dict

        path = self.cfg.output_dir / "state.pt"
        if not (self.cfg.resume and path.exists()):
            return 0
        state = torch.load(path, map_location=self.device)
        set_peft_model_state_dict(self.unet, state["lora"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        logger.logging.info("Resumed from step %d", state["step"])
        return int(state["step"])

    def _save_lora(self, name: str) -> Path:
        from diffusers import StableDiffusionPipeline
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft import get_peft_model_state_dict

        folder = self.cfg.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(folder),
            unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(self.unet)),
            safe_serialization=True,
        )
        return folder

    # ---------------------------------------------------------------- samples
    def _sample_prompts(self) -> List[dict]:
        data = json.loads(self.cfg.eval_prompts_path.read_text(encoding="utf-8"))
        self.eval_settings = data["settings"]
        seen, chosen = set(), []
        for p in data["prompts"]:  # first prompt of each category, so samples cover all behaviours
            if p["category"] not in seen:
                seen.add(p["category"])
                chosen.append(p)
        return chosen[: self.cfg.num_sample_prompts]

    @torch.no_grad()
    def _sample(self, step: int) -> None:
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        s = self.eval_settings
        pipe = StableDiffusionPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        folder = self.cfg.output_dir / "samples" / f"step_{step:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        self.unet.eval()
        for p in self._sample_prompts():
            seed = s["seeds"][0]
            gen = torch.Generator(device=self.device.type).manual_seed(seed)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                image = pipe(
                    p["prompt"], num_inference_steps=s["num_inference_steps"],
                    guidance_scale=s["guidance_scale"], generator=gen,
                    height=s["height"], width=s["width"],
                ).images[0]
            image.save(folder / f"prompt{p['id']:02d}_seed{seed}.png")
        self.unet.train()
        del pipe
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.mlflow is not None:
            self.mlflow.log_artifacts(str(folder), artifact_path=f"samples/step_{step:05d}")
        logger.logging.info("Samples saved to %s", folder)

    # ------------------------------------------------------------------ mlflow
    def _start_mlflow(self) -> None:
        self.mlflow = None
        if not self.cfg.use_mlflow:
            return
        try:
            import mlflow
        except ImportError:
            logger.logging.warning("mlflow is not installed, logging to console only")
            return
        # tracking URI and credentials come from env vars (MLFLOW_TRACKING_URI, ...) as in the MRI project
        mlflow.set_experiment(self.cfg.mlflow_experiment)
        mlflow.start_run(run_name=self.cfg.run_name)
        params = {k: str(v)[:250] for k, v in asdict(self.cfg).items()}
        mlflow.log_params(params)
        self.mlflow = mlflow

    # --------------------------------------------------------------------- run
    def run(self) -> Path:
        from diffusers.optimization import get_scheduler

        cfg = self.cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)

        self._prepare_models()
        self._prepare_data()
        self.optimizer = torch.optim.AdamW(self.params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.lr_scheduler = get_scheduler(
            cfg.lr_scheduler, optimizer=self.optimizer,
            num_warmup_steps=cfg.lr_warmup_steps, num_training_steps=cfg.max_train_steps,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        start_step = self._load_state()
        self._sample_prompts()  # also loads eval_settings (fails early if the file is missing)
        self._start_mlflow()

        self.unet.train()
        batches = self._batches()
        step = start_step
        last_sampled = -1
        try:
            while step < cfg.max_train_steps:
                step_loss = 0.0
                for _ in range(cfg.grad_accum_steps):
                    loss = self._loss(next(batches)) / cfg.grad_accum_steps
                    self.scaler.scale(loss).backward()
                    step_loss += loss.item()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1

                if self.mlflow is not None:
                    self.mlflow.log_metric("train_loss", step_loss, step=step)
                    self.mlflow.log_metric("lr", self.lr_scheduler.get_last_lr()[0], step=step)
                if step % 10 == 0 or step == 1:
                    logger.logging.info("step %d/%d loss %.4f", step, cfg.max_train_steps, step_loss)
                if step % cfg.sample_every == 0:
                    self._sample(step)
                    last_sampled = step
                if step % cfg.checkpoint_every == 0 and step < cfg.max_train_steps:
                    self._save_state(step)
                    self._save_lora(f"lora_step_{step:05d}")

            self._save_state(step)
            final = self._save_lora("lora_final")
            if last_sampled != step:  # do not generate the same samples twice
                self._sample(step)
            if self.mlflow is not None:
                self.mlflow.log_artifacts(str(final), artifact_path="lora_final")
            logger.logging.info("Training finished. LoRA weights: %s", final)
            return final
        finally:
            if self.mlflow is not None:
                self.mlflow.end_run()
