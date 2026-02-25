"""
Development rollout script for Scientist Agent.
Tests rollout inference without full training pipeline.
"""

import os
import sys

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
trainer_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(trainer_dir)

# Only add project_root to sys.path
# This ensures 'trainer' is recognized as a package (not trainer/trainer.py directly)
while project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

from trainer import Trainer, DevTaskLoader, LLM
from trainer.train_scientist.rollout_scientist import Rollout
import pandas as pd


def dev_task_loader_from_parquet(parquet_path: str = "data/val/aime24.parquet", n_samples: int = 10) -> DevTaskLoader:
    """
    Load development tasks from a parquet file.

    Args:
        parquet_path: Path to parquet file with question and result columns
        n_samples: Number of samples to load
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df = df.head(n_samples)

    if 'question' not in df.columns or 'result' not in df.columns:
        raise ValueError(f"Parquet file must have 'question' and 'result' columns. Found: {list(df.columns)}")

    ground_truth_col = 'ground_truth' if 'ground_truth' in df.columns else 'result'

    tasks = []
    for idx, row in df.iterrows():
        task = {
            "question": str(row["question"]),
            "result": str(row["result"]),
            "extra_info": {
                "groundtruth": str(row[ground_truth_col]),
                "idx": int(idx),
            }
        }
        tasks.append(task)

    return DevTaskLoader(
        tasks=tasks,
        resources={
            "main_llm": LLM(
                endpoint="https://api.openai.com/v1",
                model="gpt-4o-mini",
                sampling_parameters={"temperature": 0.7}
            ),
        },
    )


def dev_one_sample_loader() -> DevTaskLoader:
    """
    Create a single-sample dev task loader for quick testing.
    """
    question = "A point $(x,y)$ is randomly and uniformly chosen inside the square with vertices (0,0), (0,2), (2,2), and (2,0). What is the probability that $x+y < 3$?"

    task = {
        "question": question,
        "result": "0.75",
        "extra_info": {
            "groundtruth": "0.75",
            "idx": 0,
        }
    }

    tasks = [task]
    return DevTaskLoader(
        tasks=tasks,
        resources={
            "main_llm": LLM(
                endpoint="https://api.openai.com/v1",
                model="gpt-4o-mini",
                sampling_parameters={"temperature": 0.7}
            ),
        },
    )


def dev_custom_tasks_loader(tasks_list: list, model: str = "gpt-4o-mini", endpoint: str = "https://api.openai.com/v1") -> DevTaskLoader:
    """
    Create a dev task loader from custom tasks.

    Args:
        tasks_list: List of task dictionaries with keys:
            - question: str
            - result: str (expected answer)
            - extra_info: dict (optional, will be auto-generated if not provided)
        model: LLM model name
        endpoint: API endpoint

    Example:
        tasks = [
            {
                "question": "What is 2+2?",
                "result": "4"
            },
            {
                "question": "What is the capital of France?",
                "result": "Paris"
            }
        ]
        loader = dev_custom_tasks_loader(tasks)
    """
    formatted_tasks = []
    for idx, task in enumerate(tasks_list):
        if "extra_info" not in task:
            task["extra_info"] = {
                "groundtruth": task["result"],
                "idx": idx,
            }
        formatted_tasks.append(task)

    return DevTaskLoader(
        tasks=formatted_tasks,
        resources={
            "main_llm": LLM(
                endpoint=endpoint,
                model=model,
                sampling_parameters={"temperature": 0.7}
            ),
        },
    )


if __name__ == "__main__":
    """
    Run development rollout tests.

    This uses Trainer in dev mode (dev=True) which:
    - Doesn't perform training/updates
    - Just runs rollout inference
    - Saves results to rollout_data/
    """

    # Example 1: Single sample test
    print("=" * 70)
    print("Running single sample test...")
    print("=" * 70)

    # Create Rollout with scientist configuration
    rollout = Rollout(
        server_public_ip="dev_test",
        exp_name="scientist_dev_test",
        rollout_n=1,  # 1 rollout per task for dev
        batch_size=1,
        enabled_tools=["Base_Generator_Tool", "Wikipedia_RAG_Tool"],
        tool_engine=["Default", "Default"],
        module_engine=["Trainable", "Trainable", "Trainable", "Trainable"],
        max_steps=3,
        max_tokens=2048,
        train_temperature=0.7,
        test_temperature=0.0,
        output_type="direct",
        timeout=300,
    )

    # Run with Trainer in dev mode
    Trainer(n_workers=1, dev=True, max_tasks=1).fit(
        rollout,
        "http://localhost:9999/",
        dev_one_sample_loader()
    )

    print("\n" + "=" * 70)
    print("Single sample test complete!")
    print("Check rollout_data/ for results")
    print("=" * 70)

    # Example 2: Custom tasks
    # Uncomment to test custom tasks
    """
    print("\n" + "=" * 70)
    print("Running custom tasks test...")
    print("=" * 70)

    custom_tasks = [
        {
            "question": "What is the largest planet in our solar system?",
            "result": "Jupiter"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "result": "William Shakespeare"
        },
    ]

    Trainer(n_workers=1, dev=True, max_tasks=2).fit(
        rollout,
        "http://localhost:9991/",
        dev_custom_tasks_loader(custom_tasks)
    )

    print("\n" + "=" * 70)
    print("Custom tasks test complete!")
    print("=" * 70)
    """

    # Example 3: Load from parquet
    # Uncomment if you have a parquet file
    """
    print("\n" + "=" * 70)
    print("Running parquet loader test...")
    print("=" * 70)

    Trainer(n_workers=1, dev=True, max_tasks=5).fit(
        rollout,
        "http://localhost:9991/",
        dev_task_loader_from_parquet("data/val/aime24.parquet", n_samples=5)
    )

    print("\n" + "=" * 70)
    print("Parquet test complete!")
    print("=" * 70)
    """
