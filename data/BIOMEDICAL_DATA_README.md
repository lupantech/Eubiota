# Biomedical Dataset Loading Guide

This guide explains how to use the new biomedical dataset loader to replace the existing math/NQ datasets with biomedical and health-related datasets.

## Overview

The `get_biomedical_data.py` script loads and processes several large-scale biomedical and health datasets:

### Large Training Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| **MedMCQA** | ~194K | Indian medical entrance exam questions (AIIMS & NEET PG) |
| **PubMedQA Artificial** | ~211K | Biomedical research question-answering based on PubMed abstracts |
| **PubMedQA Labeled** | ~1K | High-quality labeled biomedical QA pairs |
| **MedQA (USMLE)** | ~12K | US Medical Licensing Examination-style questions |
| **MMLU Medical** | ~1.5K | Medical subjects from MMLU (anatomy, clinical knowledge, etc.) |
| **HealthSearchQA** | ~3.5K | Health-related search queries and answers |

**Total Training Data: ~420K+ questions**

### Small Validation Dataset

The validation set includes:
- MedQA validation split (~1.3K)
- PubMedQA test split (~500)
- MedMCQA validation sample (500)

**Total Validation Data: ~2.3K questions**

## Installation

Make sure you have the required packages:

```bash
pip install datasets pandas numpy fire pyarrow
```

## Usage

### Quick Start - Load All Datasets

```bash
# Load both training and validation data with default large datasets
python data/get_biomedical_data.py
```

This will create:
- `data/biomedical/combined_biomedical_train.parquet` - Combined training data
- `data/biomedical/validation_biomedical.parquet` - Validation data

### Load Only Training Data

```bash
python data/get_biomedical_data.py --mode=train
```

### Load Only Validation Data

```bash
python data/get_biomedical_data.py --mode=val
```

### Load Specific Datasets

```bash
# Load only MedMCQA and MedQA
python data/get_biomedical_data.py --datasets=medmcqa,medqa

# Available dataset options:
# - medmcqa
# - pubmedqa_artificial
# - pubmedqa_labeled
# - medqa
# - mmlu_medical
# - healthsearchqa
```

### Custom Output Directory

```bash
python data/get_biomedical_data.py --output_dir=./my_custom_data
```

## Data Format

All datasets are processed into a unified format:

```python
{
    'id': int,                    # Unique ID
    'question': str,              # The question (may include options for MCQ)
    'chain': str,                 # Empty string (for chain-of-thought, if needed)
    'result': str,                # The answer
    'source': str,                # Dataset source (e.g., 'medmcqa', 'pubmedqa')
    'extra_info': {               # Additional metadata
        'ground_truth': str,      # The correct answer
        'idx': int,               # Original index
        ...                       # Dataset-specific fields
    }
}
```

## Using with Your Training Pipeline

### Option 1: Update rollout_dev.py

Modify `train_scientist/rollout_dev.py` to use biomedical data:

```python
def dev_biomedical_loader(
    parquet_path: str = "data/biomedical/validation_biomedical.parquet",
    n_samples: int = 10
) -> DevTaskLoader:
    """Load biomedical validation tasks."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df = df.head(n_samples)

    tasks = []
    for idx, row in df.iterrows():
        task = {
            "question": str(row["question"]),
            "result": str(row["result"]),
            "extra_info": {
                "groundtruth": str(row["result"]),
                "source": str(row["source"]),
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

# Use in main:
Trainer(n_workers=1, dev=True, max_tasks=5).fit(
    rollout,
    "http://localhost:9991/",
    dev_biomedical_loader("data/biomedical/validation_biomedical.parquet", n_samples=5)
)
```

### Option 2: Full Training with Biomedical Data

For full training, update your training script to load:

```python
# Load the large training dataset
train_dataset = pd.read_parquet('data/biomedical/combined_biomedical_train.parquet')

# Load the validation dataset
val_dataset = pd.read_parquet('data/biomedical/validation_biomedical.parquet')
```

## Dataset Statistics

After loading, you'll see distribution like:

```
Dataset distribution:
  - medmcqa: 193,155 records
  - pubmedqa: 211,269 records
  - medqa: 10,178 records
  - healthsearchqa: 3,578 records

Total: ~418,180 training records
```

## Comparison with Original Datasets

| Metric | Original (Math+NQ) | New (Biomedical) |
|--------|-------------------|------------------|
| **Training Size** | ~103K | ~420K |
| **Domain** | Math + General QA | Biomedical + Health |
| **Question Types** | Math problems, factoid QA | Medical MCQ, clinical reasoning, biomedical QA |
| **Sources** | 2 datasets | 6 datasets |

## Benefits of Biomedical Datasets

1. **Domain-Specific**: All questions are biomedical/health-related
2. **Larger Scale**: 4x more training data than the original setup
3. **Diverse Sources**: Multiple high-quality biomedical datasets
4. **Real-World Relevance**: Questions from actual medical exams and research
5. **Structured Format**: Consistent schema across all datasets

## Troubleshooting

### Dataset Download Issues

If you encounter download issues:

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Or use mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### Memory Issues

If loading all datasets causes memory issues:

```bash
# Load smaller subset
python data/get_biomedical_data.py --datasets=medmcqa,medqa
```

### Custom Processing

You can modify the processing functions in `get_biomedical_data.py` to:
- Add custom fields
- Filter specific topics
- Change answer formatting
- Add data augmentation

## Next Steps

1. Run the data loader to download and process datasets
2. Test with validation set using `rollout_dev.py`
3. Update your training pipeline to use the new data
4. Monitor performance on biomedical tasks

## Example: Quick Test

```bash
# 1. Load datasets
python data/get_biomedical_data.py

# 2. Check the data
python -c "
import pandas as pd
df = pd.read_parquet('data/biomedical/validation_biomedical.parquet')
print(f'Total samples: {len(df)}')
print(f'Sources: {df.source.value_counts().to_dict()}')
print(f'\nExample question:\n{df.iloc[0].question}')
print(f'\nAnswer: {df.iloc[0].result}')
"

# 3. Test with your agent
# (Update rollout_dev.py to use biomedical data loader)
python train_scientist/rollout_dev.py
```

## Support

For issues or questions:
- Check the dataset documentation on HuggingFace
- Review the processing functions in `get_biomedical_data.py`
- Ensure all required packages are installed

---

**Note**: First run will download datasets from HuggingFace, which may take time depending on your internet connection. Subsequent runs will use cached data.
