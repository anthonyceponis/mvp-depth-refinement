#export WORKSPACE_DIR="$(dirname $0)/../.."
export WORKSPACE_DIR=$BASE_DATA_DIR../
export PYTHONPATH="$WORKSPACE_DIR":$PYTHONPATH

echo $WORKSPACE_DIR
echo $PYTHONPATH

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
macrobatch_size=1

gradient_accumulation_steps=8

accelerate launch --num_processes $num_gpus ppd_sharpdepth/training/train.py \
    --use_normal_loss \
    --sds_loss_weight 1.0 \
    --depth_weight 0.4 \
    --normal_loss_weight 0.4 \
    --base_ckpt_dir andrew-healey/sharpdepth \
    --student_ckpt_dir andrew-healey/sharpdepth \
    --add_datetime_prefix \
    --report_to wandb \
    --mixed_precision bf16 \
    --seed 42 \
    --allow_tf32 \
    --learning_rate 5e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --tracker_project_name pretrained_sharpedepth_normals \
    --wandb_name "normal loss attempt 2" \
    --set_grads_to_none \
    --checkpointing_steps 500 \
    --validation_steps 200 \
    --train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 8 \
    --use_ema \
    --base_data_dir "$WORKSPACE_DIR/data/" \
    --config "$WORKSPACE_DIR/config/train_marigold_depth_with_normals.yaml" \
    --output_dir "$WORKSPACE_DIR/train_output/" \
    --base_model unidepth \
    --denoiser lotus \
    --use_conditioning_probability 0.8 \
    --dit_patch_encoder_lr_multiplier 1 \
    --blur_unidepth_output_ratio 1 \
    --noise_aware_latent_noise_scale 0.0 \
    "$@"
