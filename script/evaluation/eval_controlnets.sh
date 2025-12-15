#!/bin/bash

set -e

cleanup() {
    echo ""
    echo "killing all child processes"
    pkill -P $$ 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

controlnet_models=(
    "edge_loss"
    "edge_loss_unidepth"
    # "no_edge_loss"
    # "no_synthetic_data"
    # "sharpdepth_loss"
    # "sds_0"
    # "sds_0.1"
    # "sds_1.0"
    # "sds_10.0"
    # "zoedepth"
    # "unidepth"
    "patchrefiner"
    "ppd_unidepth"
    "ppd_zoedepth"
    # "timestep_500_zoedepth"
    # "timestep_500_unidepth"
    "sharpdepth_lotus_zoedepth"
    "sharpdepth_lotus_unidepth"
)

declare -A model_to_model_architecture=(
    ["edge_loss"]="sharpdepth_ppd_controlnet_zoedepth"
    ["edge_loss_unidepth"]="sharpdepth_ppd_controlnet_unidepth"
    ["no_edge_loss"]="sharpdepth_ppd_controlnet_zoedepth"
    ["no_synthetic_data"]="sharpdepth_ppd_controlnet_zoedepth"
    ["sharpdepth_loss"]="sharpdepth_ppd_controlnet_zoedepth"
    ["sds_0"]="sharpdepth_ppd_timestep_500_zoedepth"
    ["sds_0.1"]="sharpdepth_ppd_timestep_500_zoedepth"
    ["sds_1.0"]="sharpdepth_ppd_timestep_500_zoedepth"
    ["sds_10.0"]="sharpdepth_ppd_timestep_500_zoedepth"
    ["unidepth"]="unidepth"
    ["zoedepth"]="zoedepth"
    ["ppd_unidepth"]="pixelperfectdepth_unidepth"
    ["ppd_zoedepth"]="pixelperfectdepth_zoedepth"
    ["timestep_500_unidepth"]="sharpdepth_ppd_timestep_500_unidepth"
    ["timestep_500_zoedepth"]="sharpdepth_ppd_timestep_500_zoedepth"
    ["patchrefiner"]="patchrefiner"
    ["sharpdepth_lotus_zoedepth"]="sharpdepth_lotus_zoedepth"
    ["sharpdepth_lotus_unidepth"]="sharpdepth_lotus_unidepth"
)

declare -A ckpt_per_model=(
    ["edge_loss"]="train_output_edge_loss/checkpoint-2500/ppd_student_controlnet/"
    ["edge_loss_unidepth"]="train_output_edge_loss_unidepth/checkpoint-2000/ppd_student_controlnet/"
    ["no_edge_loss"]="train_output_no_edge_loss/checkpoint-1500/ppd_student_controlnet/"
    ["no_synthetic_data"]="train_output_no_synthetic_data/checkpoint-2500/ppd_student_controlnet/"
    ["sharpdepth_loss"]="train_output_sharpdepth_loss/checkpoint-2500/ppd_student_controlnet/"
    ["sds_0"]="train_output_sds_0/checkpoint-1000/ppd_student/"
    ["sds_0.1"]="train_output_sds_0.1/checkpoint-1000/ppd_student/"
    ["sds_1.0"]="train_output_sds_1.0/checkpoint-1000/ppd_student/"
    ["sds_10.0"]="train_output_sds_10.0/checkpoint-1000/ppd_student/"
    ["unidepth"]="lpiccinelli/unidepth-v1-vitl14"
    ["zoedepth"]="isl-org/ZoeDepth"
    ["ppd_unidepth"]="andrew-healey/sharpdepth"
    ["ppd_zoedepth"]="andrew-healey/sharpdepth"
    ["timestep_500_unidepth"]="andrew-healey/sharpdepth"
    ["timestep_500_zoedepth"]="andrew-healey/sharpdepth"
    ["patchrefiner"]="OHo315/PatchRefiner"
    ["sharpdepth_lotus_zoedepth"]="andrew-healey/sharpdepth"
    ["sharpdepth_lotus_unidepth"]="andrew-healey/sharpdepth"
)

dataset_names=(
    "nyuv2"
    "middlebury"
    "hypersim"
)

declare -A dataset_name_to_config_path=(
    ["nyuv2"]="config/dataset_depth/data_nyu_test.yaml"
    ["middlebury"]="config/dataset_depth/data_middlebury_test.yaml"
    ["hypersim"]="config/dataset_depth/data_hypersim_test.yaml"
)

subset_size=""
selected_dataset_names=()
run_name=""
eval_only=false
debug=false
make_point_cloud=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --subset_size)
            subset_size="$2"
            shift 2
            ;;
        --dataset_names)
            IFS=',' read -ra selected_dataset_names <<< "$2"
            shift 2
            ;;
        --run_name)
            run_name="$2"
            shift 2
            ;;
        --eval_only)
            eval_only=true
            shift
            ;;
        --debug)
            debug=true
            shift
            ;;
        --make_point_cloud)
            make_point_cloud=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ ${#selected_dataset_names[@]} -eq 0 ]; then
    selected_dataset_names=("${dataset_names[@]}")
fi

if [ ${#controlnet_models[@]} -gt 8 ]; then
    echo "Error: Too many models (${#controlnet_models[@]}). Maximum is 8 (one per GPU)."
    echo "Comment out some models in the controlnet_models array before running."
    exit 1
fi

echo "=== Evaluation Configuration ==="
echo "Models: ${controlnet_models[*]}"
echo "Datasets: ${selected_dataset_names[*]}"
echo "Subset size: ${subset_size:-all}"
echo "Run name: ${run_name:-none}"
echo "Eval only: $eval_only"
echo "Debug: $debug"
echo "================================="

if [ "$eval_only" = false ]; then
    for dataset_name in "${selected_dataset_names[@]}"; do
        config_path="${dataset_name_to_config_path[$dataset_name]}"
        echo ""
        echo ">>> Running inference on dataset: $dataset_name ($config_path)"
        
        pids=()
        gpu_idx=0
        
        for model in "${controlnet_models[@]}"; do
            model_arch="${model_to_model_architecture[$model]}"
            checkpoint="${ckpt_per_model[$model]}"
            
            if [[ -z "$model_arch" ]]; then
                model_arch="$model"
            fi
            
            if [[ -z "$checkpoint" ]]; then
                checkpoint="none"
            fi
            
            cmd="CUDA_VISIBLE_DEVICES=$gpu_idx python -m ppd_sharpdepth.infer"
            cmd+=" --checkpoint $checkpoint"
            cmd+=" --dataset_config_path $config_path"
            cmd+=" --model_architecture $model_arch"
            cmd+=" --model_name $model"
            
            if [ -n "$subset_size" ]; then
                cmd+=" --subset_size $subset_size"
            fi

            if [ "$make_point_cloud" = true ]; then
                cmd+=" --make_point_cloud"
            fi
        
            
            echo "Starting inference for $model on GPU $gpu_idx..."
            eval "$cmd" &
            pids+=($!)
            
            gpu_idx=$((gpu_idx + 1))
        done
        
        echo "Waiting for all inference jobs to complete..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
        
        echo "Inference done for dataset: $dataset_name"
    done

    echo ""
    echo "=== All inference jobs completed! ==="
    echo ""
else
    echo ""
    echo "=== Skipping inference (--eval_only) ==="
    echo ""
fi

echo "=== Starting Evaluation ==="
for dataset_name in "${selected_dataset_names[@]}"; do
    config_path="${dataset_name_to_config_path[$dataset_name]}"
    
    for model in "${controlnet_models[@]}"; do
        model_arch="${model_to_model_architecture[$model]}"
        
        if [[ -z "$model_arch" ]]; then
            model_arch="$model"
        fi
        
        echo ""
        echo ">>> Evaluating $model on $dataset_name..."
        
        eval_cmd="python -m ppd_sharpdepth.eval"
        eval_cmd+=" --dataset_config_path $config_path"
        eval_cmd+=" --model_architecture $model_arch"
        eval_cmd+=" --model_name $model"
        eval_cmd+=" --run_message \"${run_name:-auto_eval}_${model}_${dataset_name}\""
        
        if [ -n "$subset_size" ]; then
            eval_cmd+=" --subset_size $subset_size"
        fi
        
        if [ "$debug" = true ]; then
            eval_cmd+=" --debug"
        fi

        eval "$eval_cmd"
        
        echo "Evaluation done for $model on $dataset_name"
    done
done

echo ""
echo "=== All evaluations completed! ==="
echo "Results saved to results.csv"
