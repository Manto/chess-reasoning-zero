export WANDB_API_KEY=5d52c7f836e8b019dacc5e60ab12e5e621d3b389
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1

# pip3 install flash-attn --no-build-isolation
# pip3 install wandb vllm==0.6.3 transformers==4.47.1

# huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir /workspace/models/Qwen2.5-1.5B-Instruct

# git clone https://github.com/volcengine/verl /workspace/verl_repo
# cd /workspace/verl_repo && pip3 install -e . -U

ray stop --force && ray start --head --dashboard-host=0.0.0.0 --include-dashboard=True

python3 -m verl.trainer.main_ppo \
 data.train_files=/workspace/data/gsm8k/train.parquet \
 data.val_files=/workspace/data/gsm8k/test.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=640 \
 data.max_prompt_length=512 \
 data.max_response_length=1024 \
 actor_rollout_ref.model.path=/workspace/models/Qwen2.5-1.5B-Instruct \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size=2 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
 critic.optim.lr=1e-5 \
 critic.model.path=/workspace/models/Qwen2.5-1.5B-Instruct \
 critic.model.enable_gradient_checkpointing=True \
 critic.model.use_remove_padding=True \
 critic.ppo_micro_batch_size=2 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=-1 \
 trainer.test_freq=100 \
 trainer.total_epochs=4 \
 trainer.project_name='qwen-countdown-grpo' \
 trainer.experiment_name="qwen_2.5_1.5b-instruct" \
 trainer.logger=['console','wandb'] 2>&1 | tee verl_demo.log