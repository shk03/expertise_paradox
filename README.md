## Overview

This repository packages the code for generating structured top‑3 brain MRI differential diagnoses via GPT‑4.1, DeepSeek‑R1 (Fireworks), or Gemini models, based on a case description.

All functionality lives under `src/llm_pipeline`, making it easy to share or extend.

## Setup

1. Create and activate a Python environment (3.10+ recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Export the `src` directory on your `PYTHONPATH` (skip if you install the package elsewhere):
   ```bash
   export PYTHONPATH=src  # or `set PYTHONPATH=src` on Windows PowerShell
   ```
4. Copy `.env.example` to `.env` and populate the required API keys:
   ```bash
   cp .env.example .env
   ```
   - `OPENAI_API_KEY` for GPT‑4.1 access.
   - `FIREWORKS_API_KEY` for DeepSeek‑R1 via Fireworks.
   - `GOOGLE_API_KEY` for Gemini.
   - Optional environment variables allow model overrides (see `.env.example`).

## CLI Usage

The CLI is exposed via `python -m llm_pipeline.cli` (or equivalently `python src/main.py`). Use `--log-level DEBUG` to see detailed progress logs.

### Generate Differential Diagnoses

```bash
python -m llm_pipeline.cli generate \
  --provider openai \
  --input data/cases.csv \
  --output data/openai_results.csv \
  --temp-output data/openai_checkpoint.csv \
  --column case_description
```

- `--provider` must be one of `openai`, `deepseek`, or `gemini`.
- `--temp-output` stores incremental generations so interrupted runs can resume.
- The input CSV must contain the case description column (default `case_description`).

## Package Layout

- `src/llm_pipeline/generation.py` – batch orchestration for the three providers.
- `src/llm_pipeline/providers/` – thin wrappers around each API.
- `src/llm_pipeline/batch.py` – resumable batching logic.

Feel free to extend `PROVIDERS` in `providers/__init__.py` to add new backends or alter renaming behaviour.
