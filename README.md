# The Expertise Paradox: Who Benefits from LLM-Assisted Brain MRI Differential Diagnosis?

Official implementation by the [HAIM Lab (Human-Centered AI in Medicine)](https://www.neurokopfzentrum.med.tum.de/neuroradiologie/forschung_projekt_haim.html) of the Technical University of Munich (TUM).
Contact: Dr. med. Su Hwan Kim (suhwan.kim@tum.de), Institute of Diagnostic and Interventional Radiology, Technical University of Munich (TUM)

## Overview

This repository packages the code for generating structured top‑3 brain MRI differential diagnoses via GPT‑4.1, DeepSeek‑R1 (Fireworks), or Gemini models, based on a case description.
All functionality lives under `src/llm_pipeline`, making it easy to share or extend.

## Study Design

<img width="698" height="302" alt="image" src="https://github.com/user-attachments/assets/d130143f-130e-4360-89f3-7f8d8185bb00" />
Readers first generated a textual description of the main imaging finding and provided their top three differential diagnoses (unassisted). Gemini 2.5 Pro, GPT-4.1, and DeepSeek-R1 were then prompted to generate their top three differential diagnoses based on a condensed medical history and each reader’s finding description. Subsequently, readers reviewed GPT-4.1’s differential diagnoses derived from their own descriptions and provided their final, integrated top three differential diagnoses (assisted). Icons were obtained from flaticon.com. <img width="468" height="114" alt="image" src="https://github.com/user-attachments/assets/564eada7-5883-4aa7-963b-b21e28bbafe7" />

## Main Findings

<img width="698" height="267" alt="image" src="https://github.com/user-attachments/assets/6a141d70-94ff-4a35-834e-715f3fc562a6" />
With increasing reader experience, absolute diagnostic LLM performance with reader-generated input improved, while relative diagnostic gains through LLM assistance paradoxically diminished. Our findings call attention to the divergence between standalone LLM performance and clinically relevant reader benefit, and emphasize the need to account for human-AI interaction in this context.<img width="468" height="114" alt="image" src="https://github.com/user-attachments/assets/e1e4b40f-8101-4f21-8f91-7df8e47b6ec3" />

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
