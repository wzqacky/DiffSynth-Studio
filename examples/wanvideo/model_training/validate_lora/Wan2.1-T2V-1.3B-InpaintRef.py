import argparse
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import MLP
from diffsynth.core import load_state_dict

def build_additional_modules(pipe, args):
    # Add default modules for accepting reference images
    # Extend patch_embedding from 16 to 33 channels
    old_pe = pipe.dit.patch_embedding
    old_weight = old_pe.weight.data
    old_bias = old_pe.bias.data
    new_in_dim = 33  # 16 noise + 16 source + 1 mask
    pipe.dit.patch_embedding = torch.nn.Conv3d(
        new_in_dim, pipe.dit.dim,
        kernel_size=tuple(pipe.dit.patch_size),
        stride=tuple(pipe.dit.patch_size)
    ).to(dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        pipe.dit.patch_embedding.weight[:, :16] = old_weight
        pipe.dit.patch_embedding.weight[:, 16:] = 0
        pipe.dit.patch_embedding.bias.copy_(old_bias)
    pipe.dit.in_dim = new_in_dim
    pipe.dit.require_vae_embedding = True

    # Add ref_conv and img_emb to the DiT
    pipe.dit.ref_conv = torch.nn.Conv2d(16, pipe.dit.dim, kernel_size=(2, 2), stride=(2, 2)).to(dtype=torch.bfloat16, device="cuda")

    clip_feat_dim = 1024 if args.image_encoder_type == "dinov2" else 1280
    pipe.dit.img_emb = MLP(clip_feat_dim, pipe.dit.dim).to(dtype=torch.bfloat16, device="cuda")
    pipe.dit.require_clip_embedding = True

    # Load LoRA weights (ignores non-LoRA keys like ref_conv.*, img_emb.*, patch_embedding.*)
    pipe.load_lora(pipe.dit, args.checkpoint, alpha=1)

    # Load ref_conv, img_emb, patch_embedding weights, and LoRA weights into the DiT
    state_dict = load_state_dict(args.checkpoint, torch_dtype=torch.bfloat16, device="cuda")

    # Load hf_encoder if hf_map is used as condition
    if args.use_hf_map and pipe.dit is not None and not hasattr(pipe.dit, 'hf_encoder'):
        from diffsynth.models.wan_video_dit import HFMapEncoder
        pipe.dit.hf_encoder = HFMapEncoder(pipe.dit.dim).to(dtype=torch.bfloat16, device="cuda")

    # Load Perceiver Resampler if needed
    if args.use_perceiver_resampler and pipe.dit is not None:
        from diffsynth.models.wan_video_dit import PerceiverResampler
        clip_feat_dim = 1024 if args.image_encoder_type == "dinov2" else 1280
        pipe.dit.perceiver_resampler = PerceiverResampler(
            clip_dim=clip_feat_dim,
            dim=pipe.dit.dim,
            num_latents=args.perceiver_num_latents,
            num_layers=args.perceiver_num_layers,
        ).to(dtype=torch.bfloat16, device="cuda")
        # Patch all CrossAttention modules so the image/text split uses num_latents
        for block in pipe.dit.blocks:
            block.cross_attn.num_image_tokens = args.perceiver_num_latents
        pipe.dit.require_clip_embedding = True

    non_lora_keys = {k: v for k, v in state_dict.items() if "lora" not in k}
    print("Non-LoRA modules:")
    print(non_lora_keys.keys())
    pipe.dit.load_state_dict(non_lora_keys, strict=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Wan2.1-T2V-1.3B customized model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint")
    parser.add_argument("--image_encoder_type", type=str, default="clip", help="Type of image encoder used")
    parser.add_argument("--val_dataset_path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--use_hf_map", action="store_true", help="Whether to use high-frequency map as condition")
    parser.add_argument("--use_perceiver_resampler", action="store_true", help="Whether to replace the vanilla MLP with a Perceiver Resampler")
    parser.add_argument("--perceiver_num_latents", type=int, default=64, help="Number of learned latent queries in the Perceiver Resampler (default: 64).")
    parser.add_argument("--perceiver_num_layers", type=int, default=4, help="Number of cross-attention layers in the Perceiver Resampler (default: 4).")
    parser.add_argument("--one_sample_only", action="store_true", help="Only validate on one sample")
    parser.add_argument("--output_dir", type=str, default="validation_results", help="Directory to save validation results")
    args = parser.parse_args()

    # Load the base pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
    )
    if args.image_encoder_type == 'dinov2':
        from diffsynth.models.dinov2_image_encoder import DINOv2ImageEncoder
        pipe.image_encoder = DINOv2ImageEncoder().to(dtype=torch.bfloat16, device="cuda")

    # Add additional modules
    build_additional_modules(pipe, args)

    # Prepare the validation dataset
    val_dataset = args.val_dataset_path

    if args.use_hf_map:
        metadata_path = f"{val_dataset}/metadata_reference_mask.csv"
    else:
        metadata_path = f"{val_dataset}/metadata.csv"

    if args.one_sample_only:
        metadata_path = str(Path(metadata_path).with_suffix("")) + "_onesample.csv"
    metadata = pd.read_csv(metadata_path)

    for idx, row in metadata.iterrows():
        prompt = row['prompt']
        if args.use_hf_map:
            if reference_mask not in row:
                raise ValueError("Using High-Frequency Maps as condition, however mask for reference image is not found.")
            reference_mask_path = f"{val_dataset}/{row['reference_mask']}"
            reference_mask = Image.open(reference_mask_path)
        else:
            reference_mask_path = None
            reference_mask = None
    
        reference_image_path = f"{val_dataset}/{row['reference_image']}"
        source_video_path = f"{val_dataset}/{row['source_video']}"
        mask_path = f"{val_dataset}/{row['mask']}"
    
        reference_image = Image.open(reference_image_path)
        source_video = VideoData(source_video_path)
        inpaint_mask = VideoData(mask_path)

        num_frames = len(source_video)
        w, h = source_video[0].size[0], source_video[0].size[1]

        video = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, artifacts",
            source_video=source_video,
            inpaint_mask=inpaint_mask,
            height=h, width=w, num_frames=num_frames,
            reference_image=reference_image,
            reference_mask = reference_mask,
            seed=1, tiled=True
        )

        output_video_path = Path(f"{args.output_dir}/{Path(args.checkpoint).parent.name}")
        output_video_path.mkdir(parents=True, exist_ok=True)
        save_video(video, f"{output_video_path}/{idx}.mp4", fps=16, quality=8)
