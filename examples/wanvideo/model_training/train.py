import torch, os, argparse, accelerate, warnings
from collections import OrderedDict
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        use_ref_conv=False,
        use_img_emb=False,
        use_inpaint_concat=False,
        image_encoder_type="clip",
        use_hf_map=False,
        use_perceiver_resampler=False,
        perceiver_num_latents=64,
        perceiver_num_layers=4,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)

        # Optionally swap CLIP image encoder with DINOv2
        if image_encoder_type == "dinov2":
            from diffsynth.models.dinov2_image_encoder import DINOv2ImageEncoder
            self.pipe.image_encoder = DINOv2ImageEncoder().to(dtype=torch.bfloat16)

        # Reference image conditioning: inject new modules into DiT
        if use_ref_conv and self.pipe.dit is not None and not hasattr(self.pipe.dit, 'ref_conv'):
            self.pipe.dit.ref_conv = torch.nn.Conv2d(
                16, self.pipe.dit.dim, kernel_size=(2, 2), stride=(2, 2)
            ).to(dtype=torch.bfloat16)
            torch.nn.init.zeros_(self.pipe.dit.ref_conv.weight)
            torch.nn.init.zeros_(self.pipe.dit.ref_conv.bias)
        if use_img_emb and self.pipe.dit is not None:
            if not hasattr(self.pipe.dit, 'img_emb'):
                from diffsynth.models.wan_video_dit import MLP
                clip_feat_dim = 1024 if image_encoder_type == "dinov2" else 1280
                self.pipe.dit.img_emb = MLP(clip_feat_dim, self.pipe.dit.dim).to(dtype=torch.bfloat16)
            self.pipe.dit.require_clip_embedding = True
        elif self.pipe.dit is not None and not hasattr(self.pipe.dit, 'img_emb'):
            self.pipe.dit.require_clip_embedding = False

        # Inpainting: extend patch_embedding from in_dim=16 to 33 (16 noise + 16 source + 1 mask)
        if use_inpaint_concat and self.pipe.dit is not None:
            old_pe = self.pipe.dit.patch_embedding
            old_weight = old_pe.weight.data   # (dim, 16, 1, 2, 2)
            old_bias = old_pe.bias.data       # (dim,)
            new_in_dim = 33  # 16 noise + 16 source + 1 mask
            self.pipe.dit.patch_embedding = torch.nn.Conv3d(
                new_in_dim, self.pipe.dit.dim,
                kernel_size=tuple(self.pipe.dit.patch_size),
                stride=tuple(self.pipe.dit.patch_size)
            ).to(dtype=torch.bfloat16)
            with torch.no_grad():
                self.pipe.dit.patch_embedding.weight[:, :16] = old_weight
                self.pipe.dit.patch_embedding.weight[:, 16:] = 0   # zero-init new channels
                self.pipe.dit.patch_embedding.bias.copy_(old_bias)
            self.pipe.dit.in_dim = new_in_dim
            self.pipe.dit.require_vae_embedding = True

        # HF-Map: pixel-space high-frequency detail encoder
        if use_hf_map and self.pipe.dit is not None and not hasattr(self.pipe.dit, 'hf_encoder'):
            from diffsynth.models.wan_video_dit import HFMapEncoder
            self.pipe.dit.hf_encoder = HFMapEncoder(self.pipe.dit.dim).to(dtype=torch.bfloat16)

        # Perceiver Resampler: replaces img_emb MLP with a learned cross-attention bottleneck
        if use_perceiver_resampler and self.pipe.dit is not None:
            from diffsynth.models.wan_video_dit import PerceiverResampler
            clip_feat_dim = 1024 if image_encoder_type == "dinov2" else 1280
            self.pipe.dit.perceiver_resampler = PerceiverResampler(
                clip_dim=clip_feat_dim,
                dim=self.pipe.dit.dim,
                num_latents=perceiver_num_latents,
                num_layers=perceiver_num_layers,
            ).to(dtype=torch.bfloat16)
            # Patch all CrossAttention modules so the image/text split uses num_latents
            for block in self.pipe.dit.blocks:
                block.cross_attn.num_image_tokens = perceiver_num_latents
            self.pipe.dit.require_clip_embedding = True

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # Unfreeze ref_conv and img_emb after switch_pipe_to_training_mode freezes everything
        if use_ref_conv and self.pipe.dit is not None and hasattr(self.pipe.dit, 'ref_conv'):
            self.pipe.dit.ref_conv.train()
            self.pipe.dit.ref_conv.requires_grad_(True)
        if use_img_emb and self.pipe.dit is not None and hasattr(self.pipe.dit, 'img_emb'):
            self.pipe.dit.img_emb.train()
            self.pipe.dit.img_emb.requires_grad_(True)
        if use_inpaint_concat and self.pipe.dit is not None:
            self.pipe.dit.patch_embedding.train()
            self.pipe.dit.patch_embedding.requires_grad_(True)
        if use_hf_map and self.pipe.dit is not None and hasattr(self.pipe.dit, 'hf_encoder'):
            self.pipe.dit.hf_encoder.train()
            self.pipe.dit.hf_encoder.requires_grad_(True)
        if use_perceiver_resampler and self.pipe.dit is not None and hasattr(self.pipe.dit, 'perceiver_resampler'):
            self.pipe.dit.perceiver_resampler.train()
            self.pipe.dit.perceiver_resampler.requires_grad_(True)

        self.print_trainable_summary()
        self.print_active_components()
        self._first_step_monitored = False

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def print_trainable_summary(self):
        groups = OrderedDict()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                group = "LoRA"
            else:
                # Use the first two dot-separated parts after "pipe." as the group name
                # e.g. "pipe.dit.ref_conv.weight" -> "dit.ref_conv"
                parts = name.replace("pipe.", "", 1).split(".")
                group = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            if group not in groups:
                groups[group] = 0
            groups[group] += param.numel()

        total = sum(groups.values())
        print("=" * 60)
        print("Trainable Parameters Summary")
        print("-" * 60)
        for group, count in groups.items():
            print(f"  {group:<30s} {count:>12,}")
        print("-" * 60)
        print(f"  {'Total':<30s} {total:>12,}")
        print("=" * 60)

    def print_active_components(self):
        print("=" * 60)
        print("Active Components (static analysis)")
        print("-" * 60)

        # Loaded models
        print("  Loaded Models:")
        for name in ["dit", "vae", "text_encoder", "image_encoder",
                      "vace", "motion_controller", "audio_encoder",
                      "animate_adapter", "vap"]:
            model = getattr(self.pipe, name, None)
            if model is not None:
                print(f"    {name}")

        # DiT configuration
        if self.pipe.dit is not None:
            print("  DiT Configuration:")
            print(f"    in_dim={self.pipe.dit.in_dim}  dim={self.pipe.dit.dim}")
            modules = []
            if hasattr(self.pipe.dit, 'ref_conv'):
                modules.append("ref_conv")
            if hasattr(self.pipe.dit, 'img_emb'):
                modules.append("img_emb")
            if hasattr(self.pipe.dit, 'hf_encoder'):
                modules.append("hf_encoder")
            if self.pipe.dit.require_vae_embedding:
                modules.append("vae_embedding (inpaint concat)")
            if self.pipe.dit.require_clip_embedding:
                modules.append("clip_embedding")
            if modules:
                print(f"    Active: {', '.join(modules)}")

        # Pipeline units
        print(f"  Pipeline Units: {len(self.pipe.units)} registered")
        print("  (unit activity will be confirmed on first training step)")
        print("=" * 60)

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "reference_mask":
                inputs_shared["reference_mask"] = data["reference_mask"][0]
            elif extra_input == "inpaint_mask":
                inputs_shared["inpaint_mask"] = data["mask"]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        monitor = not self._first_step_monitored
        if monitor:
            self._first_step_monitored = True
            print("=" * 60)
            print("Pipeline Unit Activity (first training step)")
            print("-" * 60)

        for unit in self.pipe.units:
            if monitor:
                before = set(inputs[0].keys()) | set(inputs[1].keys()) | set(inputs[2].keys())
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
            if monitor:
                after = set(inputs[0].keys()) | set(inputs[1].keys()) | set(inputs[2].keys())
                new_keys = after - before
                name = unit.__class__.__name__.replace("WanVideoUnit_", "")
                if new_keys:
                    print(f"  [ACTIVE]  {name:<30s} -> {', '.join(sorted(new_keys))}")
                else:
                    print(f"  [skip]    {name}")

        if monitor:
            print("=" * 60)

        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--use_ref_conv", default=False, action="store_true", help="Enable Approach A: add ref_conv to DiT for VAE-based reference conditioning via self-attention.")
    parser.add_argument("--use_img_emb", default=False, action="store_true", help="Enable Approach C: add img_emb to DiT for CLIP-based reference conditioning via cross-attention.")
    parser.add_argument("--use_inpaint_concat", default=False, action="store_true", help="Extend patch_embedding for inpainting: concat source video latents + mask with noise (in_dim 16->33).")
    parser.add_argument("--image_encoder_type", type=str, default="clip", choices=["clip", "dinov2"], help="Image encoder for reference conditioning: 'clip' (1280-d) or 'dinov2' (1024-d).")
    parser.add_argument("--use_hf_map", default=False, action="store_true", help="Enable HF-Map: pixel-space high-frequency detail encoder (Sobel edge map from reference image).")
    parser.add_argument("--use_perceiver_resampler", default=False, action="store_true", help="Replace img_emb MLP with a Perceiver Resampler for richer CLIP/DINOv2 cross-attention features.")
    parser.add_argument("--perceiver_num_latents", type=int, default=64, help="Number of learned latent queries in the Perceiver Resampler (default: 64).")
    parser.add_argument("--perceiver_num_layers", type=int, default=4, help="Number of cross-attention layers in the Perceiver Resampler (default: 4).")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        log_with=args.tracker,
        project_dir=args.logging_dir,
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        use_ref_conv=args.use_ref_conv,
        use_img_emb=args.use_img_emb,
        use_inpaint_concat=args.use_inpaint_concat,
        image_encoder_type=args.image_encoder_type,
        use_hf_map=args.use_hf_map,
        use_perceiver_resampler=args.use_perceiver_resampler,
        perceiver_num_latents=args.perceiver_num_latents,
        perceiver_num_layers=args.perceiver_num_layers,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
