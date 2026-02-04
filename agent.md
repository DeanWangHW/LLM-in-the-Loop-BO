# Project Analysis (LLM-in-the-Loop-BO)

## Overview
This repository implements **LLINBO** (LLM-in-the-Loop Bayesian Optimization), a hybrid framework that combines Large Language Models with statistical surrogate models (e.g., Gaussian Processes) to balance early exploration and later exploitation in black-box optimization tasks.

## Key Components
- **LLM agents**
  - `LLM_agent_BBFO.py`: LLM-assisted black-box function optimization agent.
  - `LLM_agent_HPT.py`: LLM-assisted hyperparameter tuning agent.
- **Examples / Reproducibility**
  - `BBFO_examples.ipynb`: Demonstrates the black-box optimization workflow and reproduces paper results.
  - `HPT_examples.ipynb`: Demonstrates hyperparameter tuning workflows and reproducibility steps.
  - `3D_printing_experiment.ipynb`: End-to-end 3D printing case study.
- **Utilities & Data**
  - `helper_func.py`: Shared helper routines.
  - `AM_par_func.py`: Parallel LLM-assisted BO helpers for 3D printing experiments.
  - `Black-box-opt_task_data/`, `Hyperparameter-tuning_task_data/`, `3D-printing_data/`: Task datasets and outputs.

## Typical Workflow
1. Select a task (BBFO or HPT) and open the corresponding example notebook.
2. Configure any function patterns, bounds, or loss definitions.
3. Run the notebook to reproduce results or adapt for new problems.

## External Dependencies
- OpenAI API access is used by the LLM agents (see README for setup guidance).
