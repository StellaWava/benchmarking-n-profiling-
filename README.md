# Phase 0: Benchmarking and Roofline Exploration for CPU/GPU Data Movement

This repository is a benchmarking and roofline working folder for exploring how data movement through modern machines shapes performance across scientific computing kernels.

The project is intentionally practical and reusable. It is designed to help map how different computational patterns stress the CPU and GPU in different ways, especially when the same underlying algorithmic ideas are expressed through arrays, tiles/stencils, and other structured data layouts.

## What this repository studies

The work in this folder focuses on two connected scientific-computing questions:

1. How does data movement limit performance across different machine architectures?
2. How do dependence structure and data layout change the balance between compute and memory pressure?

To explore those questions, the codebase benchmarks kernels that represent different performance motifs and dwarfs, including array-based and tile-based patterns that are commonly used in CPU and GPU workloads.

The emphasis is latency and understanding the relationship between:

- arithmetic intensity
- memory bandwidth
- synchronization and dependence depth
- roofline position
- CPU vs GPU execution behavior

## Why this folder exists

This is the Phase 0 step in a longer sprint that runs from:

Architecture → Benchmarking → LLM Dev → Finetuning

At this stage, the objective is not to produce a polished end-to-end model pipeline, but to produce grounded, reusable systems-building work that anyone can inspect, reproduce, and extend.

This means:
- benchmark kernels that make the machine constraints visible
- create clear roofline-style measurements
- test representative data movement patterns across CPU and GPU
- place the results into a common framework that can later support more advanced LLM and tuning work

## Technologies used

This repository combines multiple toolchains:

- C++ for low-level benchmark implementations
- CUDA/C++ for GPU-oriented kernels
- Python for analysis, plotting, and roofline-style post-processing
- CMake-based build organization for compiled benchmarks

The code is intended to make it easy to compare different data movement regimes and expose how structure in the algorithm affects the machine-level bottleneck.

## Core benchmark themes

The current work is centered on the following benchmark themes:

- CPU and GPU roofline-style performance measurement
- streaming and bandwidth-driven kernels
- compute-bound and memory-bound behavior comparison
- analysis of data movement patterns across arrays, blocks, and tiled formulations
- mapping the same algorithmic ideas across different machine layers

This gives the repository a useful bridge between low-level systems understanding and later higher-level model-development work.

## Repository structure

- `cpp/` — benchmark and kernel source files in C++ and CUDA/C++
- `python/` — analysis and plotting scripts
- `build/` — generated build outputs and local CMake artifacts
- `roofline.csv`, `roofline1.csv` — measurement snapshots used for roofline-style analysis
- `roofline_chart.png` — generated visualization output

## Public access and cloning

This repository is public and can be cloned directly with:

```bash
git clone https://github.com/StellaWava/benchmarking-n-profiling-.git
cd benchmarking-n-profiling-
```

If you are using this workspace directly, the project root is already the benchmark working folder and can be explored from there.

## Getting started

A standard workflow for using the repository is:

1. Clone the repository
2. Inspect the benchmark sources in `cpp/`
3. Build the relevant benchmark target with CMake/Ninja
4. Run the executable(s) to collect performance data
5. Use the Python analysis scripts to plot and interpret the results

## Reuse and extension

This repository is structured to be reusable for people who want to:

- understand roofline-style benchmarking in practice
- compare CPU and GPU behavior on representative kernels
- study how data layout and dependence structure affect performance
- carry the same measurement contract into later LLM-related work

## Project framing

This work is inspired by the broader question raised in the Cost of Distance blog: how do we build mental models for knowledge convergence in computing by tracing how enduring constraints reappear across systems, architectures, and models? Read blog here: https://substack.com/@thecostofdistance/note/c-298245222?utm_source=notes-share-action&r=30yfjj 

The benchmark folder is a concrete way to operationalize that question. It is not just a code dump or a single isolated benchmark. It is a grounded first step in a longer trajectory from architectural understanding to benchmarking, then to model development and finetuning.

## Phase 0 summary

This repository represents Phase 0 of the long sprint:
Architecture → Benchmarking → LLM Dev → Finetuning

The value of this phase is that it produces reusable, measurable, and explainable systems work that can be shared publicly and carried forward into later stages.

## Suggested next steps

1. Capture a clean roofline data table for the current machine
2. Label each kernel by its dependence structure and expected dwarf class
3. Re-run the benchmark with consistent configuration and reproducibility notes
4. Extend the same measurement contract to later LLM and finetuning components
