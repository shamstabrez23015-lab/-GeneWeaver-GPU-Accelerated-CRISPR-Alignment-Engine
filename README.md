# GeneWeaver
### GPU-Accelerated CRISPR Alignment Engine

GeneWeaver is a high-performance bioinformatics project designed to detect potential CRISPR off-target DNA mutations using GPU-accelerated sequence alignment.

It uses custom CUDA kernels to perform parallel DNA sequence comparisons, reducing computational time and enabling efficient genomic analysis.

## Technology Stack

- Python
- BioPython – Genomic FASTA parsing
- Numba CUDA – GPU-accelerated computation
- Dask – Distributed workload scheduling
- Textual – Terminal-based user interface

## Key Modules

1. Genomic Parser
2. CUDA Kernel Engine
3. Distributed Scheduler
4. TUI Dashboard

## Project Domain

Bioinformatics & High-Performance Computing (HPC)

## Project Objective

To build a fast, scalable, and scientifically accurate CRISPR alignment engine capable of identifying potential off-target mutations through GPU parallel processing.