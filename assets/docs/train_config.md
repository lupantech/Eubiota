# Training Configuration Guide

The training process for Science Agent is controlled by a central configuration file located at `trainer/train_scientist/config.yaml`. This document details the key parameters and their usage.

## 1. Environment & Compute Settings

These settings control hardware usage and basic experiment metadata.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WANDB_ENTITY` | `sci-agent` | Weights & Biases entity/team name for logging. |
| `CUDA_VISIBLE_DEVICES`| `0,1,2,3` | GPU IDs to be used. |
| `N_GPUS` | `4` | Total number of GPUs available for the experiment. |
| `N_WORKERS` | `12` | Number of ray workers for parallel data processing/rollouts. |
| `EXPERIMENT_NAME` | `scientist_8b_...` | Name of the experiment (used for logging and checkpoints). |
| `PROJECT_NAME` | `Scientist_...` | Project name grouping multiple experiments. |
| `BASE_DATA_DIR` | `/PATH_TO_...` | **Crucial**: Absolute path to the data directory. Modify this to matches your system path. |

## 2. Model Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_MODEL` | `Qwen/Qwen3-8B` | The backbone LLM to be trained (e.g., Qwen, Llama). Serves as the base for rollouts. If not found locally, it will be pulled from Hugging Face. |
| `ROLLOUT_TP_SIZE` | `1` | Tensor Parallelism size for rollout generation. Keep at 1 for smaller models (7B/8B). |
| `module_engines` | `["Trainable", ...]` | Specifies which agent modules are trained vs. fixed. See [Multi-Agent Architecture](#multi-agent-architecture). |

### Multi-Agent Architecture (`MODULE_ENGINE`)
The `MODULE_ENGINE` list defines the backend for the four core agents in order:
1.  **Planner**
2.  **Executor**
3.  **Verifier**
4.  **Generator**

Values can be:
*   `"Trainable"`: This module's weights will be updated during RL training (uses `BASE_MODEL`).
*   `"gpt-4o"` / `"dashscope"`: Uses a fixed external API (frozen).
*   `"vllm-..."`: Uses a fixed locall hosted model.

**Example**:
```yaml
# Train only Planner, use GPT-4o for others
MODULE_ENGINE: ["Trainable", "gpt-4o", "gpt-4o", "gpt-4o"]
```

## 3. Tool Configuration

| Parameter | Description |
|-----------|-------------|
| `ENABLE_TOOLS` | List of tool class names to activate (e.g., `Google_Search_Tool`, `PubMed_Search_Tool`). |
| `TOOL_ENGINE` | Corresponding backend for each tool. `None` usually implies deterministic tools (code), while `gpt-4o` implies an LLM-driven tool wrapper. |

## 4. Training Hyperparameters (`python_args`)

These arguments are passed directly to the underlying training script (VeRL/PPO).
For more detailed parameter understanding, please refer to [VeRL Documentation](https://verl.readthedocs.io/en/latest/examples/config.html).

### Data & Rollout
*   `data.train_files`: Path to training data (parquet).
*   `data.val_files`: Path to validation data.
*   `data.train_batch_size`: Global batch size.
*   `actor_rollout_ref.rollout.n`: Number of rollouts generated per prompt.
*   `data.max_prompt_length`: Max input tokens (default: 18432).
*   `data.max_response_length`: Max output tokens (default: 2048).

### PPO Parameters
*   `actor_rollout_ref.actor.optim.lr`: Learning rate (default: `1e-6`).
*   `actor_rollout_ref.actor.ppo_mini_batch_size`: Mini-batch size for PPO updates.
*   `actor_rollout_ref.actor.use_kl_loss`: Enable KL penalty (True/False).
*   `actor_rollout_ref.actor.kl_loss_coef`: Coefficient for KL penalty (default: `0.001`).

### Training Logistics
*   `trainer.total_epochs`: Number of full training passes (default: `5`).
*   `trainer.save_freq`: Checkpoint saving frequency (every N epochs).
*   `trainer.test_freq`: Evaluation frequency.
*   `trainer.val_before_train`: Run validation before starting training (useful for baselines).

## 5. Agent Behavior

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOOL_STEPS` | `3` | Maximum number of tool execution rounds per turn. Warning: increasing >5 may overflow context. |
| `TRAIN_TEMPERATURE` | `0.7` | Temperature for sampling during training rollouts. |
| `TEST_TEMPERATURE` | `0.0` | Temperature for evaluation (greedy decoding). |
| `AGENT_MAX_TIMEOUT`| `500` | Max time (seconds) allowed for a single agent step. |

---

**Note**: To apply changes, edit `trainer/train_scientist/config.yaml` directly before launching the training script.
