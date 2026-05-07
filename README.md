# lab21_2A202600344

Short project repository for the lab assignment 2A202600344. This workspace contains the experiment notebook, adapter files, and result summaries used for evaluating adapted models and ranking experiments.

## Contents
- `notebook.ipynb` — Jupyter notebook with the main experiments and analysis.
- `adapters/` — Adapter artifacts used in experiments.
	- `adapters/r16/` — Adapter configuration and model weights.
- `results/` — CSV summaries and qualitative comparison outputs.

## Quick Start
1. Open the notebook `notebook.ipynb` in Jupyter or JupyterLab.
2. Follow the notebook cells to load adapters from `adapters/`, run evaluations, and generate results saved into `results/`.

## Important files
- `adapters/r16/adapter_config.json` — Adapter configuration.
- `adapters/r16/adapter_model.safetensors` — Trained adapter weights.
- `results/qualitative_comparison.csv` — Qualitative comparison outputs.
- `results/rank_experiment_summary.csv` — Ranking experiment summary.

## Notes
- This repository is focused on reproducible notebook-based experiments; use the notebook to reproduce results step-by-step.
- If you need the environment details (Python packages, versions), consider adding a `requirements.txt` or `environment.yml` to the workspace.

If you want, I can expand this README with setup commands, dependency lists, or example outputs — tell me which you'd like added.