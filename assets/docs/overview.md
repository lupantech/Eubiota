# Overview

Eubiota is a **modular agentic platform** driving end-to-end discovery in the human microbiome. By unifying multi-agent reasoning with a suite of domain-specific tools, the system bridges the gap between computational hypothesis and experimental reality. From interactive exploration to high-throughput screening, Eubiota translates fragmented biological data into actionable, wet-lab validated discoveries.

## Key Features

- **Multi-Agent Reasoning** — Decomposing complex scientific inquiries into reliable, multi-step reasoning workflows. Four coordinated agents—Planner, Executor, Verifier, and Generator—collaborate via shared memory to decompose goals, validate tool outputs, and ensure robust, long-horizon reasoning.

- **Tool-Grounded Discovery** — Unifying biological databases, literature search, and computation into an iterative reasoning loop. Native access to PubMed, KEGG, MDIPID, and 55+ lab protocols. Every conclusion is explicitly grounded in retrieved evidence and verifiable execution traces, not just LLM parametric knowledge.

- **End-to-End Validation** — Driving the full scientific loop, from large-scale hypothesis generation to wet-lab verification. Validated across four end-to-end studies: gene prioritization, therapeutic consortia design, antibiotic optimization, and metabolite discovery—all confirmed via experimental assays.

- **Scalable Deployment** — Scaling from interactive chat to high-throughput, configuration-driven discovery engines. Supports interactive chat for human-in-the-loop oversight, configuration workflows for parallelized screening, and fully secure local deployment to ensure clinical data sovereignty.


## Architecture

A modular framework that decouples scientific inquiry into specialized agents for planning, execution, verification, and grounded generation.

### Modules

| Module | Description |
|--------|-------------|
| **Planner** | Decomposes inquiries into executable subgoals. Optimized via agentic reinforcement learning to ensure robust, long-horizon scientific reasoning. |
| **Executor** | Translates plans into validated tool commands. Enforces strict interface constraints to ensure reliable execution across diverse APIs. |
| **Verifier** | Evaluates evidence for consistency and completeness, acting as a gatekeeper to prevent hallucinations and determine termination. |
| **Generator** | Synthesizes the final answer strictly grounded in the execution trace, ensuring every claim is traceable to retrieved sources. |
| **Memory** | Maintains a structured shared state of the full reasoning history, enabling seamless inter-agent handoffs and full workflow auditability. |
| **Toolset** | Integrates 18+ domain-specific tools spanning PubMed, KEGG, MDIPID, and lab protocols to bridge computational and wet-lab reasoning. |

### Multi-Agent Reasoning Loop

Specialized agents orchestrate an iterative cycle of planning, execution, and verification via shared memory to ensure rigorous evidence grounding. Eubiota iteratively cycles through this RL-optimized loop, self-correcting strategies and accumulating evidence to produce a verifiable, citation-backed conclusion.

### Integrated Tools

Eubiota integrates a diverse set of **18+ domain-specific tools** spanning literature retrieval, biological databases, web search, gene analysis, and laboratory resources. Its modular design enables seamless expansion—researchers can plug in new tools, proprietary datasets, or custom analysis modules via a unified interface.

| Category | Tools | Description |
|----------|-------|-------------|
| **Knowledge Bases** | KEGG Disease/Drug/Gene/Organism, MDIPID Disease/Gene/Microbe, Gene Phenotype Mapping, Protocol Database (55+ protocols) | Native access to authoritative biological and biomedical databases for structured knowledge retrieval |
| **Literature Search** | PubMed, Google, Perplexity, Wikipedia, URL Context Extraction | Comprehensive literature and web search for evidence gathering and citation grounding |
| **Computational** | Python Code Generator, Base Generator, Document Context Search, Database Context Search | Code execution and advanced context analysis for complex reasoning tasks |
