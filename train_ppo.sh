# This script trains a PPO model on the Countdown dataset using the Qwen2.5-3B model as the base model.
# Configured for 2x A100 80GB GPUs

export N_GPUS=3
export CUDA_VISIBLE_DEVICES=0,1,2
ray stop --force && ray start --head --dashboard-host=0.0.0.0 --include-dashboard=True
export BASE_MODEL="Qwen/Qwen2.5-3B"
export DATA_DIR="data/countdown"
export ROLLOUT_TP_SIZE=3
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=81 \
    data.val_batch_size=640 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    +compute_score_path=trainer.reward_score.scoring_fn \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=3 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=6 \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name=chess-r1-zero-countdown \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log