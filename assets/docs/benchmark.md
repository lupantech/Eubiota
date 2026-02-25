# Benchmark Curation and Evaluation

To evaluate Eubiota, we use a dual-component evaluation framework that combines established biomedical benchmarks with a domain-specific benchmark focused on host–microbiome–drug reasoning.

## General Biomedical Benchmarks

We use community benchmarks to validate foundational biomedical competence and multi-step reasoning.

| Benchmark | Subset / Focus | What it measures | Metric |
|----------|-----------------|------------------|--------|
| **MedMCQA** | Medicine subject subset | Professional clinical knowledge and diagnostic-style reasoning | Accuracy vs. gold labels |
| **WMDP-Bio** | Dual-use bio / expert-level biology | High-stakes biological knowledge (e.g., virology, immunology, epidemiology) and expert reasoning | Accuracy vs. gold labels |

## Domain-Specific Benchmark (MDIPID-Derived)

To assess specialized capabilities for pharmacomicrobiomics and mechanistic microbiome reasoning, we curate a benchmark derived from the **MDIPID** database. The suite is structured as multiple-choice tasks with gold labels and focuses on the following targeted reasoning skills:

| Task | Name | Core capability evaluated |
|------|------|---------------------------|
| **Task 1** | **Drug–Microbe Impact (Drug-Imp)** | Identify taxa with directional abundance changes (enrichment/depletion) in response to drugs or dietary interventions |
| **Task 2** | **Microbe–Protein Mechanism (MB-Mec)** | Pinpoint enzymes/proteins that mechanistically mediate microbe–drug interactions |
| **Task 3** | **Protein Functional Comprehension (Prot-Func)** | Select the correct protein function within a given microbial species among close biochemical distractors |
| **Task 4** | **Protein–Gene Mapping (Prot-Gen)** | Map functional protein descriptions to standardized gene names for genomic grounding |

## Evaluation Protocol

All tasks are formatted as multiple-choice questions. We use **GPT-4o** as an automated evaluator with a standardized two-step grading procedure:

1. **Answer extraction**: parse the final model response to extract the selected option (independent of any accompanying rationale).
2. **Scoring**: compare the extracted option against the gold label and report accuracy.

To reduce sensitivity to minor formatting variations, we apply **text-similarity matching** to map extracted outputs to the nearest valid option when needed.

