#!/bin/bash
# Wan2.1-T2V-1.3B Inpainting with Reference Image LoRA

# Activate conda environment
source /data/Anaconda3/etc/profile.d/conda.sh
conda activate diffsynth

export CUDA_VISIBLE_DEVICES="6"
export HF_ENDPOINT="https://hf-mirror.com/"

# Dataset paths
DATASET_BASE=data/inpaint_dataset_resized
METADATA_CSV=data/inpaint_dataset_resized/metadata_reference_mask.csv # use the csv with reference mask path added

# Training parameters
LEARNING_RATE=1e-4
NUM_EPOCHS=5
LORA_RANK=32
GRADIENT_ACCUM=4

# Output path
MODEL_NAME="Wan2.1-T2V-1.3B-InpaintRef-hfmap-dinov2"
OUTPUT_PATH="./models/train/${MODEL_NAME}"

echo "=========================================="
echo "Wan2.1-T2V-1.3B Inpainting with Reference Image Training"
echo "=========================================="
echo "Model: Wan2.1-T2V-1.3B"
echo "Dataset: ${DATASET_BASE}"
echo "Metadata: ${METADATA_CSV}"
echo "Output: ${OUTPUT_PATH}"
echo "Python: $(which python)"
echo "=========================================="

accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE}" \
  --dataset_metadata_path "${METADATA_CSV}" \
  --data_file_keys "video,source_video,mask,reference_image,reference_mask" \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path ${OUTPUT_PATH} \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank ${LORA_RANK} \
  --extra_inputs "source_video,inpaint_mask,reference_image,reference_mask" \
  --use_hf_map \
  --use_inpaint_concat \
  --use_ref_conv \
  --use_img_emb \
  --image_encoder_type dinov2 \
  --use_gradient_checkpointing_offload \
  --tracker "tensorboard" \
  --tracker_project_name "${MODEL_NAME}" \
  --logging_dir ${OUTPUT_PATH}/logs \

echo "Training complete! Output saved to: ${OUTPUT_PATH}"