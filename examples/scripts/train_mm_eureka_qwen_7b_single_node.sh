set -x

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export NCCL_TIMEOUT=7200

export DATASET="data/k12.jsonl"


OUTPUT_DIR='K12_onlinefilter'
PRETRAIN_MODEL="Qwen2-5-VL-7B-Instruct"
export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
cd MM-EUREKA-Qwen
export WORKING_DIR=MM-EUREKA-Qwen
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
    --working-dir $WORKING_DIR \
    --runtime-env MM-EUREKA-Qwen/runtime_env.json \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url examples/scripts/reward_func_qwen_instruct.py \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.3 \
   --vllm_sync_backend nccl \
   --pretrain $PRETRAIN_MODEL \
   --save_path ${OUTPUT_DIR} \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 128 \
   --temperature 1.0 \
   --n_samples_per_prompt 8 \
   --lambd 1.0 \
   --gamma 1.0 \
   --max_epochs 1 \
   --num_episodes 5 \
   --prompt_max_len 3000 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --advantage_estimator group_norm \
   --zero_stage 1 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.0 \
   --prompt_data $DATASET \
   --disable_fast_tokenizer \
   --input_key message \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 50 \
   --ckpt_path "${OUTPUT_DIR}/ckpt" \
   --max_ckpt_num 1000000 \
   --save_hf_ckpt \
   --freeze_prefix visual \
   --enable_accuracy_filter \
   --accuracy_lower_bound 0.1 \
   --accuracy_upper_bound 0.9 \
   --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
   --load_checkpoint | tee ${OUTPUT_DIR}/training.log

ray stop