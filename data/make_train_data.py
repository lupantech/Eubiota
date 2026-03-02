"""
Make Training Data Script
Consolidated script for creating training and validation datasets.

Features:
- Downloads microbio data from Eubiota/Microbiome-Reasoning
- Loads math (DeepMath-103K), search (NQ), and general bio (pubmedqa, medqa) datasets
- Mixes with ratio math:search:general_bio:microbio = 1:2:1:6 (default)
- Uses ALL microbio data as anchor to control proportions
- Creates train and val datasets

Usage:
    # Default ratio 1:2:1:6
    python make_train_data.py

    # Custom ratio
    python make_train_data.py --ratio 1 2 1 6

    # Only create train or val
    python make_train_data.py --mode train
    python make_train_data.py --mode val
"""

import json
import os
import argparse
import pandas as pd
import numpy as np
import datasets
from datasets import Dataset, concatenate_datasets


# ============================================================================
# Helper Functions
# ============================================================================

def process_answers(answers, to_string=True):
    """
    Processes answer fields and returns a STRING or list.
    Handles: list, tuple, numpy array, string, number, None, etc.
    """
    items = []

    if isinstance(answers, np.ndarray):
        items = [str(item) for item in answers.flatten() if item is not None and pd.notna(item)]
    elif isinstance(answers, (list, tuple)):
        items = [str(item) for item in answers if item is not None and pd.notna(item)]
    elif isinstance(answers, str):
        cleaned = answers.strip()
        if cleaned:
            items = [cleaned]
    elif isinstance(answers, (int, float, np.generic)):
        if not pd.isna(answers):
            items = [str(answers).strip()]
    elif answers is None:
        items = []
    else:
        s = str(answers).strip()
        if s and s != "nan":
            items = [s]

    if to_string:
        return "; ".join(items) if items else ""
    else:
        return items


# ============================================================================
# Dataset Processing Functions
# ============================================================================

def process_nq_dataset(dataset):
    """Processes the NQ dataset to a unified schema."""
    processed_data = []
    for idx, item in enumerate(dataset):
        question = item.get("question", "").strip()
        if question and not question.endswith('?'):
            question += '?'

        golden_answers = item.get("golden_answers", [])
        final_result = process_answers(golden_answers, to_string=True)

        new_entry = {
            'id': idx,
            'question': question,
            'chain': "",
            'result': str(final_result),
            'source': "nq",
            'extra_info': {
                'ground_truth': str(final_result),
                'idx': idx
            }
        }
        processed_data.append(new_entry)

    df = pd.DataFrame(processed_data)
    return Dataset.from_pandas(df, preserve_index=False)


def process_math_dataset(dataset):
    """Processes the DeepMath-103K dataset to a unified schema."""
    def map_fn(example, idx):
        question = example.pop('question') if 'question' in example else example.pop('Problem')
        solution = example.pop('final_answer') if 'final_answer' in example else example.pop('Answer')

        return {
            "id": idx,
            "question": question,
            "chain": "",
            "result": str(solution),
            "source": "mathhard",
            "extra_info": {
                'ground_truth': str(solution),
                'idx': idx,
            }
        }

    return dataset.map(function=map_fn, with_indices=True, remove_columns=dataset.column_names)


def process_pubmedqa_dataset(dataset):
    """Processes PubMedQA dataset."""
    processed_data = []

    for idx, item in enumerate(dataset):
        question = item.get('question', '').strip()

        context_dict = item.get('context', {})
        contexts = context_dict.get('contexts', []) if isinstance(context_dict, dict) else []
        context_text = ' '.join(contexts) if contexts else ''

        if context_text:
            full_question = f"Context: {context_text[:500]}...\n\nQuestion: {question}"
        else:
            full_question = question

        final_decision = item.get('final_decision', '')
        long_answer = item.get('long_answer', '')
        answer = final_decision if final_decision else long_answer

        new_entry = {
            'id': idx,
            'question': full_question,
            'chain': '',
            'result': str(answer),
            'source': 'pubmedqa',
            'extra_info': {
                'ground_truth': str(answer),
                'final_decision': final_decision,
                'pubid': item.get('pubid', ''),
                'idx': idx
            }
        }
        processed_data.append(new_entry)

    df = pd.DataFrame(processed_data)
    return Dataset.from_pandas(df, preserve_index=False)


def process_medqa_dataset(dataset):
    """Processes MedQA (USMLE) dataset (~12K questions)."""
    processed_data = []

    for idx, item in enumerate(dataset):
        question = item.get('question', '').strip()

        options = item.get('options', {})
        if isinstance(options, dict):
            options_text = '\n'.join([f"{k}. {v}" for k, v in options.items()])
            full_question = f"{question}\n{options_text}"
        else:
            full_question = question

        answer_idx = item.get('answer_idx', '')
        answer = item.get('answer', '')
        result = f"{answer_idx}. {answer}" if answer_idx else answer

        new_entry = {
            'id': idx,
            'question': full_question,
            'chain': '',
            'result': str(result),
            'source': 'medqa',
            'extra_info': {
                'ground_truth': str(result),
                'answer_idx': answer_idx,
                'meta_info': item.get('meta_info', ''),
                'idx': idx
            }
        }
        processed_data.append(new_entry)

    df = pd.DataFrame(processed_data)
    return Dataset.from_pandas(df, preserve_index=False)


def process_microbio_dataset(dataset):
    """Processes microbio dataset to ensure unified schema."""
    processed_data = []

    for idx, item in enumerate(dataset):
        raw_meta = item.get('metadata', '{}')
        extra_info = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
        new_entry = {
            'id': idx,
            'question': item.get('question', ''),
            'chain': '',
            'result': item.get('ground_truth', ''),
            'source': 'Synthesized MicroBio',
            'extra_info': extra_info,
        }
        processed_data.append(new_entry)

    df = pd.DataFrame(processed_data)
    return Dataset.from_pandas(df, preserve_index=False)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_microbio_train_data():
    """Load microbio training data from Eubiota/Microbiome-Reasoning."""
    print("\n" + "="*70)
    print("Loading MicroBio training data from Eubiota/Microbiome-Reasoning...")
    print("="*70)

    try:
        microbio = datasets.load_dataset(
            'Eubiota/Microbiome-Reasoning',
            split='train'
        )
        processed = process_microbio_dataset(microbio)
        print(f"Loaded {len(processed)} records from MicroBio train")
        return processed
    except Exception as e:
        print(f"Failed to load MicroBio train data: {e}")
        return None


def load_microbio_test_data(n_samples=100):
    """Load microbio test data from Eubiota/Microbiome-Reasoning."""
    print("\n" + "="*70)
    print(f"Loading MicroBio test data ({n_samples} samples)...")
    print("="*70)

    try:
        microbio = datasets.load_dataset(
            'Eubiota/Microbiome-Reasoning',
            split='test'
        )
        # Sample n_samples
        if len(microbio) > n_samples:
            microbio = microbio.shuffle(seed=42).select(range(n_samples))
        processed = process_microbio_dataset(microbio)
        print(f"Loaded {len(processed)} records from MicroBio test")
        return processed
    except Exception as e:
        print(f"Failed to load MicroBio test data: {e}")
        return None


def load_math_data():
    """Load Math data from DeepMath-103K."""
    print("\n" + "="*70)
    print("Loading Math data from DeepMath-103K...")
    print("="*70)

    try:
        math_dataset = datasets.load_dataset('zwhe99/DeepMath-103K', split='train')
        processed = process_math_dataset(math_dataset)
        print(f"Loaded {len(processed)} records from Math")
        return processed
    except Exception as e:
        print(f"Failed to load Math data: {e}")
        return None


def load_search_data():
    """Load Search data from NQ dataset."""
    print("\n" + "="*70)
    print("Loading Search data from NQ dataset...")
    print("="*70)

    try:
        nq_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq', split='train')
        processed = process_nq_dataset(nq_dataset)
        print(f"Loaded {len(processed)} records from NQ")
        return processed
    except Exception as e:
        print(f"Failed to load NQ data: {e}")
        return None


def load_general_bio_data(n_samples=None):
    """
    Load general biomedical datasets: pubmedqa and medqa (split evenly).

    Args:
        n_samples: Total number of samples needed. If provided, will sample
                   n_samples/2 from each dataset. If None, loads all data.
    """
    print("\n" + "="*70)
    print("Loading General Bio datasets (PubMedQA + MedQA, split evenly)...")
    print("="*70)

    pubmedqa_data = None
    medqa_data = None

    # 1. PubMedQA
    try:
        print("\nLoading PubMedQA...")
        pubmedqa = datasets.load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')
        pubmedqa_data = process_pubmedqa_dataset(pubmedqa)
        print(f"Loaded {len(pubmedqa_data)} records from PubMedQA")
    except Exception as e:
        print(f"Failed to load PubMedQA: {e}")

    # 2. MedQA
    try:
        print("\nLoading MedQA...")
        medqa = datasets.load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
        medqa_data = process_medqa_dataset(medqa)
        print(f"Loaded {len(medqa_data)} records from MedQA")
    except Exception as e:
        print(f"Failed to load MedQA: {e}")

    if pubmedqa_data is None and medqa_data is None:
        raise ValueError("No general bio datasets were successfully loaded!")

    # Sample evenly from each dataset if n_samples is specified
    if n_samples is not None and n_samples > 0:
        samples_per_dataset = n_samples // 2

        sampled_datasets = []

        if pubmedqa_data is not None:
            n_pubmed = min(samples_per_dataset, len(pubmedqa_data))
            pubmedqa_sampled = pubmedqa_data.shuffle(seed=42).select(range(n_pubmed))
            sampled_datasets.append(pubmedqa_sampled)
            print(f"Sampled {n_pubmed} from PubMedQA")

        if medqa_data is not None:
            n_medqa = min(samples_per_dataset, len(medqa_data))
            medqa_sampled = medqa_data.shuffle(seed=42).select(range(n_medqa))
            sampled_datasets.append(medqa_sampled)
            print(f"Sampled {n_medqa} from MedQA")

        combined = concatenate_datasets(sampled_datasets)
    else:
        # Combine all data
        datasets_to_combine = []
        if pubmedqa_data is not None:
            datasets_to_combine.append(pubmedqa_data)
        if medqa_data is not None:
            datasets_to_combine.append(medqa_data)
        combined = concatenate_datasets(datasets_to_combine)

    print(f"\nTotal general bio records: {len(combined)}")

    return combined


# ============================================================================
# Main Data Creation Functions
# ============================================================================

def create_train_data(output_dir='data/train', ratio=(1, 2, 1, 6)):
    """
    Create balanced training dataset with ratio math:search:general_bio:microbio.

    Uses ALL microbio data as anchor to control proportions.
    """
    math_ratio, search_ratio, bio_ratio, microbio_ratio = ratio

    print("\n" + "="*70)
    print(f"Creating training dataset with ratio {math_ratio}:{search_ratio}:{bio_ratio}:{microbio_ratio}")
    print("(math:search:general_bio:microbio)")
    print("="*70)

    # 1. Load microbio data first to calculate sample sizes
    microbio_data = load_microbio_train_data()
    if microbio_data is None:
        raise ValueError("Failed to load microbio data! Please remember to access at https://huggingface.co/datasets/Eubiota/Microbiome-Reasoning")

    df_microbio = microbio_data.to_pandas()

    # 2. Calculate sample sizes based on microbio (use ALL microbio data)
    n_microbio = len(df_microbio)
    base_size = n_microbio // microbio_ratio

    n_math = base_size * math_ratio
    n_search = base_size * search_ratio
    n_bio = base_size * bio_ratio

    print("\n" + "="*70)
    print(f"Using ALL {n_microbio} microbiome samples as anchor")
    print(f"Base size: {base_size}")
    print(f"Target - Math: {n_math}, Search: {n_search}, Bio: {n_bio}")
    print("="*70)

    # 3. Load other datasets
    math_data = load_math_data()
    search_data = load_search_data()
    # Load bio data with n_samples for even split between pubmedqa and medqa
    bio_data = load_general_bio_data(n_samples=n_bio)

    # Convert to pandas for easier manipulation
    df_math = math_data.to_pandas() if math_data else pd.DataFrame()
    df_search = search_data.to_pandas() if search_data else pd.DataFrame()
    df_bio = bio_data.to_pandas() if bio_data else pd.DataFrame()

    print(f"\nAvailable - Math: {len(df_math)}, Search: {len(df_search)}, Bio: {len(df_bio)}")

    # 4. Sample from math and search categories
    if n_math > len(df_math):
        print(f"Warning: Need {n_math} math samples but only have {len(df_math)}")
        n_math = len(df_math)

    if n_search > len(df_search):
        print(f"Warning: Need {n_search} search samples but only have {len(df_search)}")
        n_search = len(df_search)

    df_math_sample = df_math.sample(n=n_math, random_state=42) if n_math < len(df_math) else df_math
    df_search_sample = df_search.sample(n=n_search, random_state=42) if n_search < len(df_search) else df_search
    # Bio is already sampled evenly in load_general_bio_data
    df_bio_sample = df_bio

    print(f"\nSampling:")
    print(f"  Math: {len(df_math_sample)}")
    print(f"  Search: {len(df_search_sample)}")
    print(f"  Bio: {len(df_bio_sample)} (evenly split between PubMedQA and MedQA)")
    print(f"  MicroBio: {len(df_microbio)} (all)")

    # 4. Combine and shuffle
    df_combined = pd.concat([df_math_sample, df_search_sample, df_bio_sample, df_microbio], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    df_combined['id'] = range(len(df_combined))

    # Reorder columns to match original format
    df_combined = df_combined[['question', 'id', 'chain', 'result', 'source', 'extra_info']]

    # 5. Show distribution
    print("\n" + "="*70)
    print("Final dataset distribution:")
    print("="*70)
    source_counts = df_combined['source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count:,} records")
    print(f"\nTotal records: {len(df_combined):,}")

    # 6. Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'train.parquet')
    df_combined.to_parquet(output_file, index=False)
    print(f"\nSaved to {output_file}")

    return df_combined


def create_val_data(output_dir='data/val', n_samples=100):
    """
    Create validation dataset from microbio test data.

    Args:
        output_dir: Output directory
        n_samples: Number of samples to extract (default: 100)
    """
    print("\n" + "="*70)
    print(f"Creating validation dataset ({n_samples} samples)")
    print("="*70)

    # Load microbio test data
    val_data = load_microbio_test_data(n_samples=n_samples)
    if val_data is None:
        raise ValueError("Failed to load microbio test data!")

    df_val = val_data.to_pandas()
    df_val['id'] = range(len(df_val))

    # Reorder columns to match original format
    df_val = df_val[['question', 'id', 'chain', 'result', 'source', 'extra_info']]

    print(f"\nValidation dataset: {len(df_val)} records")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'val.parquet')
    df_val.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")

    return df_val


def main():
    parser = argparse.ArgumentParser(
        description='Create training and validation datasets'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'train', 'val'],
        help='Mode: all (default), train, or val'
    )
    parser.add_argument(
        '--ratio',
        type=int,
        nargs=4,
        default=[1, 2, 1, 6],
        metavar=('MATH', 'SEARCH', 'BIO', 'MICROBIO'),
        help='Ratio for math:search:general_bio:microbio (default: 1 2 1 6)'
    )
    parser.add_argument(
        '--train-output',
        type=str,
        default='data/train',
        help='Output directory for training data (default: data/train)'
    )
    parser.add_argument(
        '--val-output',
        type=str,
        default='data/val',
        help='Output directory for validation data (default: data/val)'
    )
    parser.add_argument(
        '--val-samples',
        type=int,
        default=100,
        help='Number of validation samples (default: 100)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Ratio (math:search:bio:microbio): {args.ratio[0]}:{args.ratio[1]}:{args.ratio[2]}:{args.ratio[3]}")
    print(f"  Train output: {args.train_output}")
    print(f"  Val output: {args.val_output}")
    print(f"  Val samples: {args.val_samples}")
    print("="*70)

    if args.mode in ['all', 'train']:
        create_train_data(output_dir=args.train_output, ratio=tuple(args.ratio))

    if args.mode in ['all', 'val']:
        create_val_data(output_dir=args.val_output, n_samples=args.val_samples)

    print("\n" + "="*70)
    print("All done!")
    print("="*70)


if __name__ == '__main__':
    main()
