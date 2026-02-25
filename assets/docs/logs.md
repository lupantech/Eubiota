## Training Logs and Outputs

We provide comprehensive logs to monitor training. 

### Training Logs

During training, training output logs are automatically saved with IP-based organization:
```
task_logs/
└── {PUBLIC_IP}/
    └── train_log/
        ├── training_output_0000  # First 1MB
        ├── training_output_0001  # Next 1MB
        └── ...
```

**Configuration:**
- Logs are split into 1MB files for easier management (configurable in `train/train_with_logs.sh`)
- Maximum 5000 log files retained
- Monitor latest logs: `tail -f task_logs/{YOUR_IP}/train_log/training_output_*`
### Rollout Data

During training, rollout trajectories are saved for analysis:
```
rollout_data/
└── {PUBLIC_IP}/
    └── {EXPERIMENT_NAME}_{TIMESTAMP}/     # e.g., rollout_all_7B_{time_stamp}
        ├── .init.lock
        ├── .run_info
        └── {MODEL_NAME}_{TIMESTAMP}/      # e.g., Qwen2.5-7B-Instruct_{time_stamp}
            ├── train/                      # Training rollouts (usually empty to save space)
            └── validation/
                ├── .val.lock
                └── step_0/                 # Validation at global step 0
                    ├── idx_0/              # Individual validation samples
                    │   └── rollout_{uuid}.json
                    ├── idx_1/
                    └── ...
```

**Note:** Step numbers restart from 0 after each training restart.

**Rollout JSON Fields:**
| Field | Description |
|-------|-------------|
| `prompt` | Original problem/query |
| `groundtruth` | Expected answer |
| `answer_extracted` | Model's predicted answer |
| `reward` | Score (0.0 = incorrect, positive = correct) |
| `total_result` | Full execution trace with:<br>• `query_analysis`: Problem breakdown<br>• `memory`: Tool execution history<br>• `direct_output`: Final response<br>• Tool prompts and responses |
| `timestamp` | Generation time |


### Model Checkpoints

**Directory Structure:**
```
checkpoints/
└── {PROJECT_NAME}/           # e.g., Eubiota (from config.yaml)
    └── {EXPERIMENT_NAME}/    # e.g., rollout_all_7B_useklloss (from config.yaml)
        ├── global_step_2/
        │   ├── actor/
        │   │   └── huggingface/  # HuggingFace format (ready for inference)
        │   └── data.pt           # Training state
        ├── global_step_4/
        ├── global_step_6/
        └── latest_checkpointed_iteration.txt  # Points to latest checkpoint
```

**Configuration (`train/config.yaml`):**
- `trainer.save_freq`: Save interval (default: every 2 epochs)
- `trainer.test_freq`: Validation interval (default: every 2 epochs)
- `trainer.total_epochs`: Total epochs (default: 5)

**Usage:**
- **VLLM inference**: Configure paths in `scripts/serve_vllm.sh`
- **Direct loading**: `transformers.from_pretrained("checkpoints/{PROJECT}/{EXPERIMENT}/global_step_X/actor/huggingface/")`

