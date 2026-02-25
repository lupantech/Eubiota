# Dataset Preparation & Customization

This document describes the datasets used for training Science Agent and how to customize the training data mix.

## 1. Dataset Overview

We construct our training data by mixing four distinct domains to foster diverse reasoning capabilities:

### A. Domain-Specific Benchmark (Microbiome Reasoning)
To evaluate specialized capabilities in dissecting complex host–microbiome–drug interactions, we use a curated domain-specific benchmark derived from the MDIPID database. This suite comprises four targeted reasoning tasks:

*   **Task 1: Drug–Microbe Impact (Drug-Imp)**
    *   **Description**: Identifies specific bacterial taxa that exhibit directional changes (enrichment or depletion) in response to pharmaceutical agents or dietary interventions.
    *   **Goal**: Evaluates fine-grained understanding of pharmacomicrobiomics and directional effects in the gut environment.

*   **Task 2: Microbe–Protein Mechanism (MB-Mec)**
    *   **Description**: Pinpoints specific enzymes or functional proteins responsible for a microbe's metabolic interaction with a drug.
    *   **Goal**: Tests higher-order mechanistic reasoning by bridging organism-level phenotypes and specific molecular drivers.

*   **Task 3: Protein Functional Comprehension (Prot-Func)**
    *   **Description**: Identifies the precise biological function of a protein within a specific microbial species.
    *   **Goal**: Evaluates the ability to resolve species-specific functional ambiguities by distinguishing correct activity from closely related biochemical distractors.

*   **Task 4: Protein–Gene Mapping (Prot-Gen)**
    *   **Description**: Maps functional protein descriptions to standardized gene names in the corresponding organism.
    *   **Goal**: Validates precision in genomic grounding, essential for downstream genetic engineering and experimental design.

### B. General Medical-Biology Reasoning
*   **Datasets**: [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) & [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
*   **Purpose**: Provides broad medical knowledge and clinical reasoning capabilities.

### C. Mathematical Reasoning
*   **Dataset**: [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
*   **Purpose**: Enhances the model's ability to perform logical deduction and numerical reasoning.

### D. Agentic Search
*   **Dataset**: [NQ (Natural Questions)](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets)
*   **Purpose**: Trains the model to handle information retrieval and question answering from open-domain sources.

---

## 2. Generating Training Data

The script `data/make_train_data.py` is used to download, process, and mix these datasets into a unified training file.

### Default Usage
The functionality mixes datasets with a default ratio of **Math : Search : Bio : MicroBio = 1 : 2 : 1 : 6**.

```bash
python data/make_train_data.py
```

This command will:
1.  Load all specialized Microbiome data (this serves as the "anchor" for sample size).
2.  Calculate the number of samples needed for other domains based on the ratio.
3.  Sample from Math, Search, and Bio datasets.
4.  Combine and shuffle the data.
5.  Save the result to `data/train/train.parquet`.

### Customizing Ratios
You can customize the mixing ratio using the `--ratio` argument. The order is: `MATH SEARCH BIO MICROBIO`.

**Example: Equal distribution (1:1:1:1)**
```bash
python data/make_train_data.py --ratio 1 1 1 1
```

**Example: Heavy focus on Medical/Bio (1:1:4:4)**
```bash
python data/make_train_data.py --ratio 1 1 4 4
```

### Other Options
*   **`--mode`**: Select operation mode.
    *   `all` (default): Generate both training and validation sets.
    *   `train`: Generate only training set.
    *   `val`: Generate only validation set (from MicroBio test split).
*   **`--val-samples`**: Number of samples for the validation set (default: 100).

```bash
# Only generate validation set with 200 samples
python data/make_train_data.py --mode val --val-samples 200
```

---

## 3. How to Customize Datasets

The `data/make_train_data.py` script is designed to be extensible. To add a new dataset or modify an existing one, follow these steps:

### A. Add a Loading Function
Create a function to load your dataset (e.g., from HuggingFace or a local file).

```python
def load_my_new_dataset():
    dataset = datasets.load_dataset('my_org/my_dataset', split='train')
    return process_my_dataset(dataset)
```

### B. Add a Processing Function
Create a function to map your dataset's fields to the unified schema:
*   `id`: Unique identifier
*   `question`: The input query
*   `result`: The target answer
*   `chain`: (Optional) Chain of thought
*   `source`: Source name (for tracking)
*   `extra_info`: (Optional) Any metadata

```python
def process_my_dataset(dataset):
    # logic to rename/format columns
    ...
    return processed_dataset
```

### C. Update Mixing Logic
In `create_train_data()`:
1.  Add a ratio parameter for your new dataset.
2.  Call your load function.
3.  Calculate sample size based on the ratio.
4.  Add your dataset to `pd.concat`.
