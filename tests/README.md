# Benchmark Evaluation Suite

This folder contains a comprehensive benchmark evaluation framework for testing and evaluating the Science Agent across multiple scientific datasets and models.

## 📋 Overview

The evaluation suite allows you to:
- Run scientific agent evaluations on multiple benchmark datasets
- Support multiple LLM backends (OpenAI API, Dashscope Qwen, local vLLM servers)
- Batch process questions with configurable tools and modules
- Calculate and aggregate performance metrics across datasets
- Generate detailed logs and results for analysis

## 📁 Folder Structure

### Core Scripts
- **`solve.py`** - Main solver class that runs the agent on individual questions with configurable parameters (max steps, max time, output formats)
- **`calculate_score_unified.py`** - Unified scoring module that evaluates agent responses and extracts answers using LLM-based verification
- **`utils.py`** - Utility functions for analyzing results, calculating tool usage statistics, and processing execution logs

### Benchmark Datasets
Each dataset folder contains:
- `data/data.json` - Question dataset with standardized format (question, choices, expected answer)

**Available Datasets:**
- `bomixqa/` - Biomedical question answering
- `gaia/` - General AI benchmark
- `lab-bench-litqa2/` - Literature question answering (Part 2)
- `lab-bench-protocolqa/` - Protocol-based questions
- `mmlu-bio/` - Medical/Biology subset of MMLU
- `pubmedqa/` - PubMed-based QA
- `science_gene/` - Gene-related scientific questions
- `wmdp-bio/` - Biological knowledge protection benchmark
- `Drug-Microbiome_Impact/` - Drug-Microbiome impact analysis
- `Drug-Microbiome_Interaction_Type/` - Drug-Microbiome interaction classification
- `MB-Protein_Molecular_Mechanism/` - Protein molecular mechanism
- `Protein_Genotype_Phenotype_Mapping/` - Genotype-Phenotype association

### Experiment Scripts
- `exp/run_all_models_all_datasets.sh` - Main batch evaluation script supporting parallel processing
- `exp/serve_vllm.sh` - Helper script for starting local vLLM servers

## 🚀 Quick Start

### 1. Run Evaluation on a Single Dataset

```bash
python solve.py \
    --agent gpt-4o \
    --task lab-bench-litqa2 \
    --index 0
```

### 2. Run Batch Evaluation (All Datasets, Multiple Models)

```bash
bash exp/run_all_models_all_datasets.sh
```

This script supports:
- Parallel processing with configurable thread count
- Multiple model backends (OpenAI, Dashscope, local vLLM)
- Multiple tool/module configurations
- Automatic result aggregation

### 3. Calculate Evaluation Scores

```bash
python calculate_score_unified.py \
    --result_dir results/GPT4o-test-merge \
    --output_file final_scores.json
```

## ⚙️ Configuration

### Model Definitions
Models are configured in `exp/run_all_models_all_datasets.sh` with format:
```
[port:]modelname,label,enabled_tools,tool_engines,module_engines
```

Examples:
- **OpenAI API**: `gpt-4o,GPT4o-test,Protocol_Search_Tool|Pubmed_Search_Tool,...`
- **Dashscope Qwen**: `dashscope-qwen2.5-7b,QwenMax-Bio,...`
- **Local vLLM**: `8000:/path/to/model,LocalModel,Protocol_Search_Tool|...`

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--agent` | LLM model to use (e.g., gpt-4o, dashscope-qwen2.5-7b) |
| `--task` | Dataset/benchmark name |
| `--max_steps` | Maximum steps for agent to solve problem (default: 10) |
| `--max_time` | Maximum execution time in seconds (default: 60) |
| `--max_tokens` | Maximum tokens per response (default: 4000) |
| `--output_types` | Output format: 'direct', 'final', or 'schema' (comma-separated) |
| `--temperature` | Sampling temperature (default: 0.7) |
| `--add_tag` | Add formatting tags to prompts (default: False) |

## 📊 Output Formats

### Direct Output
Raw response from the agent as-is.

### Final Output
Structured output with explicit final answer extraction.

### Schema Output
Validated output following a defined schema format.

## 📈 Metrics Calculated

- **Accuracy**: Percentage of correct answers
- **Average Steps**: Mean number of steps taken per question
- **Average Time**: Mean execution time per question
- **One-Step Rate**: Percentage of questions solved in single step
- **Tool Usage**: Frequency distribution of tool usage across problems

## 🔧 Advanced Usage

### Running with Local vLLM Server

1. Start vLLM server:
```bash
bash exp/serve_vllm.sh
```

2. Configure model in batch script with port specification:
```bash
"8000:/path/to/model,LocalModel,Protocol_Search_Tool|...,gpt-4o|...,Trainable|gpt-4o|..."
```

### Custom Tool Configuration

Enable specific tools by modifying the `enabled_tools` list in the model definition. Available tools include:
- Search tools: Protocol_Search_Tool, Pubmed_Search_Tool, Google_Search_Tool, etc.
- Database tools: KEGG_*, MDIPID_*, Gene_Phenotype_Search_Tool, etc.
- Execution tools: Python_Coder_Tool, Database_Context_Search_Tool
- Generation tools: Base_Generator_Tool


## 🤝 Contributing

When adding new datasets:
1. Create a new folder with dataset name
2. Add `data/data.json` following the standard format
3. Update `exp/run_all_models_all_datasets.sh` to include the new task in the `TASKS` list
4. Run evaluation and verify results are generated correctly

