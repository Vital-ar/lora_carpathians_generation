"""Smoke test for the LoRA training and image generation code with tiny random models (CPU, no download).

It checks the code path: dataset -> training steps -> checkpoint -> resume ->
sample generation -> LoRA weights saved -> loading the LoRA and generating the
evaluation images (with skip_existing). It does NOT check image quality.

Run from the project root:  python -m tests.smoke_test_lora
"""
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from src.entity.generate_entity import GenerateEntity
from src.entity.lora_entity import LoraTrainEntity
from src.pipe.generate_images_pipe import run_generate_images_pipe
from src.pipe.lora_train_pipe import run_lora_train_pipe
from src.logging import logger

def bytes_to_unicode() -> dict:
    """GPT-2 / CLIP byte-to-unicode table (copied so the test does not depend on a private import)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def make_tiny_models(tmp: Path) -> dict:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    chars = list(bytes_to_unicode().values())
    tokens = chars + [c + "</w>" for c in chars] + ["<|startoftext|>", "<|endoftext|>"]
    vocab = {t: i for i, t in enumerate(tokens)}
    (tmp / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(tmp / "vocab.json"), str(tmp / "merges.txt"), model_max_length=16)

    torch.manual_seed(0)
    text_encoder = CLIPTextModel(CLIPTextConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=16, projection_dim=32,
        bos_token_id=len(vocab) - 2, eos_token_id=len(vocab) - 1,
    ))
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=4,
        block_out_channels=(8, 8), down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2, sample_size=32,
    )
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1, norm_num_groups=4,
        block_out_channels=(8, 16), cross_attention_dim=32, attention_head_dim=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    )
    return {
        "tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae, "unet": unet,
        "noise_scheduler": DDPMScheduler(num_train_timesteps=100),
    }


def make_tiny_pipe(tmp: Path, lora_dir: Path = None):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    m = make_tiny_models(tmp)
    pipe = StableDiffusionPipeline(
        vae=m["vae"], text_encoder=m["text_encoder"], tokenizer=m["tokenizer"], unet=m["unet"],
        scheduler=DPMSolverMultistepScheduler.from_config(m["noise_scheduler"].config),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
    )
    if lora_dir is not None:
        pipe.load_lora_weights(str(lora_dir))
    return pipe


def make_data(tmp: Path) -> tuple:
    train = tmp / "train"
    train.mkdir()
    with open(train / "metadata.jsonl", "w", encoding="utf-8") as f:
        for i in range(4):
            Image.new("RGB", (32, 32), (40 * i, 80, 120)).save(train / f"img{i}.jpg")
            f.write(json.dumps({"file_name": f"img{i}.jpg", "text": f"crpth, scene number {i}"}) + "\n")
    prompts = tmp / "eval_prompts.json"
    prompts.write_text(json.dumps({
        "settings": {"trigger": "crpth", "width": 32, "height": 32, "num_inference_steps": 2,
                     "guidance_scale": 3.0, "negative_prompt": "", "seeds": [1]},
        "prompts": [
            {"id": 1, "category": "in_domain", "prompt": "crpth, a meadow"},
            {"id": 2, "category": "unrelated", "prompt": "a cat"},
        ],
    }), encoding="utf-8")
    return train, prompts


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        train, prompts = make_data(tmp)
        out = tmp / "run"

        def cfg(steps: int) -> LoraTrainEntity:
            return LoraTrainEntity(
                resolution=32, rank=4, max_train_steps=steps, batch_size=2, grad_accum_steps=2,
                checkpoint_every=2, sample_every=2, num_sample_prompts=2, num_workers=0,
                use_mlflow=False, mixed_precision=False, gradient_checkpointing=True,
                run_name="smoke", train_dir=train, eval_prompts_path=prompts, output_dir=out,
            )

        run_lora_train_pipe(cfg(2), models=make_tiny_models(tmp))            # first run: 2 steps
        assert (out / "state.pt").exists(), "state.pt missing"
        run_lora_train_pipe(cfg(4), models=make_tiny_models(tmp))            # resumes at step 2 -> 4
        final = out / "lora_final"
        files = sorted(p.name for p in final.iterdir())
        assert any(n.endswith(".safetensors") for n in files), f"no weights saved: {files}"
        n_samples = len(list((out / "samples").rglob("*.png")))
        assert n_samples >= 2, f"samples missing ({n_samples})"

        # generation: baseline (no LoRA) and fine-tuned (with LoRA, scale 0.5), then resume with skip_existing
        for name, lora, scale in (("base", None, 1.0), ("tuned", final, 0.5)):
            gen_cfg = GenerateEntity(run_name=name, lora_path=lora, lora_scale=scale,
                                     eval_prompts_path=prompts, output_dir=tmp / "gen" / name)
            manifest = run_generate_images_pipe(gen_cfg, pipe=make_tiny_pipe(tmp, lora))
            info = json.loads(manifest.read_text(encoding="utf-8"))
            pngs = list((tmp / "gen" / name).rglob("*.png"))
            assert len(info["images"]) == 2 and len(pngs) == 2, f"{name}: {len(pngs)} images"
        before = {p: p.stat().st_mtime_ns for p in (tmp / "gen" / "tuned").rglob("*.png")}
        run_generate_images_pipe(gen_cfg, pipe=make_tiny_pipe(tmp, final))
        after = {p: p.stat().st_mtime_ns for p in (tmp / "gen" / "tuned").rglob("*.png")}
        assert before == after, "skip_existing did not skip"

        print("SMOKE TEST PASSED:", files, f"| {n_samples} sample images | generation OK")


if __name__ == "__main__":
    main()