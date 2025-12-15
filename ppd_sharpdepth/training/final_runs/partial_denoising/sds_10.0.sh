export WORKSPACE_DIR=$BASE_DATA_DIR../
export PYTHONPATH="$WORKSPACE_DIR":$PYTHONPATH

num_gpus=$NUM_GPUS
macrobatch_size=1

gradient_accumulation_steps=$((macrobatch_size / num_gpus))

    # --student_ckpt_dir_revision depth_anything_small_run \
    # --student_ckpt_dir_revision blurred \
    # --student_ckpt_dir_revision identity_no_sds \
accelerate launch --num_processes $num_gpus ppd_sharpdepth/training/train.py \
    --sds_loss_weight 10.0 \
    --depth_weight 1.6 \
    --base_ckpt_dir andrew-healey/sharpdepth \
    --student_ckpt_dir andrew-healey/sharpdepth \
    --add_datetime_prefix \
    --report_to wandb \
    --mixed_precision bf16 \
    --seed 42 \
    --allow_tf32 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --tracker_project_name ppd_sharpdepth_train \
    --wandb_name "sds_10.0" \
    --set_grads_to_none \
    --checkpointing_steps 1000 \
    --validation_steps 200 \
    --train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --num_train_epochs 1 \
    --use_ema \
    --base_data_dir "$WORKSPACE_DIR/data/" \
    --config "$WORKSPACE_DIR/config/train_marigold_depth.yaml" \
    --output_dir "$WORKSPACE_DIR/train_output_sds_10.0/" \
    --base_model zoedepth \
    --denoiser pixel_perfect_depth \
    --use_conditioning_probability 0.8 \
    --dit_patch_encoder_lr_multiplier 1 \
    --blur_unidepth_output_ratio 32 \
    --noise_aware_latent_noise_scale 0.0 \
    --use_conditioning_for_initial_ppd \
    --initialize_ppd_from_timestep 500 \
    --max_sds_timestep 500 \
    "$@"