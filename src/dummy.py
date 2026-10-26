import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from diffusers import AutoencoderKL, DDPMScheduler 
from transformers import CLIPTextModel

class LoRADiffuserDataset(Dataset):
    def __init__(self, jsonl_file_path, images_dir, tokenizer_name_or_path="runwayml/stable-diffusion-v1-5", size=512, center_crop=False, image_key="crpth", prompt_key="prompt"):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, subfolder="tokenizer")
        
        # Read lines from the JSONL metadata file
        self.entries = []
        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line.strip()))
                    
        # Image Preprocessing Transforms (normalized to [-1, 1] for VAE encoding)
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        image_filename = entry[self.image_key]
        caption = entry[self.prompt_key]
        
        # Load image
        image_path = self.images_dir / image_filename
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        # Transform pixel values
        pixel_values = self.image_transforms(image)
        
        # Tokenize caption prompt
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0) # Squeeze batch dimension for PyTorch DataLoader collate
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

def collate_fn(examples):
    input_ids = [example["input_ids"] for example in examples]
    pixel_values = [example["pixel_values"] for example in examples]

    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack(input_ids)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    # Include class images/prompts if prior preservation is active
    if "class_pixel_values" in examples:
        class_pixel_values = [example["class_pixel_values"] for example in examples]
        class_input_ids = [example["class_input_ids"] for example in examples]
        
        batch["class_pixel_values"] = torch.stack(class_pixel_values).to(memory_format=torch.contiguous_format).float()
        batch["class_input_ids"] = torch.stack(class_input_ids)

    return batch

# ----------------- Model Setup & Training Preparation ----------------- #

model_id = "runwayml/stable-diffusion-v1-5"

# Load the U-Net component
unet = UNet2DConditionModel.from_pretrained(
    model_id,
    subfolder="unet",
    torch_dtype=torch.float32
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"]
)

# Wrap U-Net with PEFT to attach trainable adapter layers
unet = get_peft_model(unet, lora_config)

# Verify only LoRA parameters are trainable
unet.print_trainable_parameters()

# Production optimizer targeting trainable LoRA parameters
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-2)

# Cosine learning rate scheduler with warmup
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=5000)

# Enable gradient checkpointing for VRAM optimization
unet.enable_gradient_checkpointing()

# ----------------- Construct DataLoader ----------------- #

# Note: Instantiate your dataset here before creating the DataLoader
dataset = LoRADiffuserDataset(jsonl_file_path="data.jsonl", images_dir="images/")

train_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
)

# ----------------- Save and Inference Example ----------------- #

unet.save_pretrained("my-custom-sd15-lora")

# Deployment/Inference: Load base pipeline and inject LoRA weights
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# Inject LoRA adapter
pipeline.load_lora_weights("my-custom-sd15-lora")

# Generate output image
# image = pipeline("your prompt here").images[0]

model_id = "runwayml/stable-diffusion-v1-5" 
device = "cuda" if torch.cuda.is_available() else "cpu" 
# Load frozen VAE, Text Encoder, and Noise Scheduler 
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device) 
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device) 
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler") 
vae.requires_grad_(False) 
text_encoder.requires_grad_(False)


def prepare_latents_and_conditioning(batch, vae, text_encoder, noise_scheduler, device): 
    pixel_values = batch["pixel\_values"].to(device) 
    input_ids = batch["input_ids"].to(device) 
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample() 
        latents = latents * vae.config.scaling_factor
        encoder_hidden_states = text_encoder(input_ids)

    noise = torch.randn_like(latents) 
    bsz = latents.shape
    timesteps = torch.randint( 0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device ).long()
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) 
    return noisy_latents, timesteps, encoder_hidden_states, noise
