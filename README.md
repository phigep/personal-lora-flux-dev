## LoRA Evaluation for FLUX.1-dev (Person-Specific)

<a href="Introduction_to_Low_Rank_Adaptation_of_Generative_Diffusion_Models_Geppner.pdf">Open the project report (PDF)</a>

<object data="Introduction_to_Low_Rank_Adaptation_of_Generative_Diffusion_Models_Geppner.pdf" type="application/pdf" width="100%" height="640" style="border: 1px solid #ddd; border-radius: 6px;">
  <p>Your viewer does not support embedded PDFs. <a href="Introduction_to_Low_Rank_Adaptation_of_Generative_Diffusion_Models_Geppner.pdf">Download the PDF</a>.</p>
</object>

---

### Overview
This repository provides scripts, configs, and notebooks to train and evaluate person-specific LoRA adapters for the FLUX.1-dev diffusion model. It evaluates generated images using LLM-based metrics (GPT-4o) and CLIP-based semantic alignment, and includes analyses on resolution, LoRA rank, frequency artifacts, and training-step effects.

- **Training**: example LoRA scripts and dataset configs for several variants (resolution, rank, caption strategies)
- **Evaluation**: LLM-based metrics plus CLIP score; batch evaluation helper
- **Analysis**: notebooks and plots comparing variants and highlighting artifacts
- **Results**: consolidated CSVs for quick inspection

If you’re browsing results, see the CSVs at the project root (e.g., `res_florence.csv`, `res_florence1024.csv`, `res_florencer16.csv`, `res_gpt4o.csv`, `res_keyword.csv`) and figures like `r4_results_llm.png`, `r16_results_llm.png`, `frequ_r4_r32.png`.

### Environment
- **Python**: 3.11 (see `.python-version`)
- **Dependencies** (see `pyproject.toml`): jupyter, ipykernel, ipywidgets, matplotlib, seaborn, plotnine, numpy, pandas, scikit-image, scikit-learn, scikit-misc, torch, torchmetrics[image,multimodal], opencv-python, python-dotenv, openai, tqdm

#### Setup using uv (recommended)
```bash
# Install uv if needed
pip install uv

# From the repo root
uv sync

# Activate the environment if uv created .venv
source .venv/bin/activate
```

#### Setup using pip (alternative)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install \
  ipykernel ipywidgets jupyter matplotlib openai opencv-python pandas plotnine \
  pydantic python-dotenv scikit-image scikit-learn scikit-misc seaborn \
  torch "torchmetrics[image,multimodal]" tqdm
```

#### API Keys
LLM-based evaluation uses OpenAI (model: `gpt-4o`). Create a `.env` in the repo root:
```bash
OPENAI_API_KEY=your_key_here
```
The evaluation code uses `python-dotenv` to load environment variables.

### Repository Layout (selected)
- `eval/`
  - `eval.py`: evaluation helpers (LLM-based metrics, CLIP score) and `test_all_scores(...)` for batch evaluation
- `loras/`
  - `personal-lora-fluxdev-florence2/`
  - `personal-lora-fluxdev-florence2-1024/`
  - `personal-lora-fluxdev-florence2-r16/`
  - `personal-lora-fluxdev-gpt4ocaptions/`
  - `personal-lora-fluxdev-keyword/`
  - Each contains `train.sh` (example command), `dataset.toml` (data config), and sample outputs/prompts
- **Notebooks**: `eval_loras.ipynb`, `eval/test_eval.ipynb`, `resolution_comparison.ipynb`, `rank_comparison.ipynb`, `frequency.ipynb`, `magnitude_artifacts.ipynb`, `images_captioning_experiment/*`
- **Results & media**: CSVs (`res_*.csv`), plots (`r4_results_llm.png`, `r16_results_llm.png`, `frequ_r4_r32.png`), and `magnitude_artifacts_training_steps_count.csv`
- **Data examples**: `input_images/`, `metric_eval_test_images/`

Note: A nested `lora_eval_Geppner/` directory mirrors much of the top-level. Treat the top-level as canonical unless you specifically need the nested copy.

### Training (LoRA)
Example commands are in `loras/*/train.sh`. They reference environment-specific absolute paths; update them for your system before running.

Key points from `dataset.toml`:
- `class_tokens` uses the custom token `<phigep>` for person-specific adaptation
- Example resolution: `512`
- Rank/resolution variants (e.g., `-r16`, `-1024`) can be trained by adjusting network dimension, dataset resolution, and output names

Run a script after paths are edited:
```bash
bash loras/personal-lora-fluxdev-florence2/train.sh
```

### Evaluation
Core utilities are in `eval/eval.py`.

- **LLM-based metrics** (model `gpt-4o`):
  - `eval_person_llm` (0/1 person present)
  - `eval_same_person_llm` (0–1 identity consistency vs base image)
  - `eval_prompt_adherence_llm` and `eval_prompt_adherence_llm_simpler_prompt` (0–1 prompt adherence)
  - `eval_image_quality_llm` (0–1 visual quality)
  - `eval_limb_quality_llm` (0–1 plausibility of limbs/proportions)
  - `eval_artifacts_quality_llm` (0–1; higher means fewer artifacts)
- **CLIP score** using `openai/clip-vit-base-patch16` via `torchmetrics`

Batch evaluation helper:
- `test_all_scores(foldername, single_param_funcs, two_param_funcs, three_param_funcs, base_image_path, prompt_dict)`
  - Expects image filenames like `loraname_trainingstep_promptid_timestamp.png`
  - Returns a pandas DataFrame of scores

Minimal example:
```python
from eval.eval import (
    test_all_scores,
    eval_person_llm, eval_image_quality_llm, eval_limb_quality_llm,
    eval_artifacts_quality_llm, eval_same_person_llm,
    eval_prompt_adherence_llm, eval_prompt_adherence_llm_simpler_prompt,
)

prompt_dict = {
    "00": "<phigep>",
    "01": "<phigep> sitting on a bench in central park wearing a black hoodie and pink crocs",
    "02": "<phigep> playing chess in space",
}

df = test_all_scores(
    foldername="path/to/generated_images",
    single_param_funcs=[
        eval_person_llm,
        eval_image_quality_llm,
        eval_limb_quality_llm,
        eval_artifacts_quality_llm,
    ],
    two_param_funcs=[eval_same_person_llm],
    three_param_funcs=[
        eval_prompt_adherence_llm,
        eval_prompt_adherence_llm_simpler_prompt,
    ],
    base_image_path="metric_eval_test_images/p1.jpg",
    prompt_dict=prompt_dict,
)

df.to_csv("results.csv", index=False)
```

### Results at a Glance
- `res_florence.csv`: Florence captions at 512px
- `res_florence1024.csv`: 1024px training variant
- `res_florencer16.csv`: rank-16 LoRA variant
- `res_gpt4o.csv`: GPT-4o captions variant
- `res_keyword.csv`: keyword-caption variant

Each CSV has identifiers (`loraname`, `training_step`, `prompt_id`, `timestamp`), LLM metrics (person, same-person, prompt adherence variants, image quality, limb quality, artifacts), and `clip_score`.

Included figures: `r4_results_llm.png`, `r16_results_llm.png` (metric trends) and `frequ_r4_r32.png` (frequency analysis). See `magnitude_artifacts_training_steps_count.csv` for simple artifact counts by step.

### Notes and Caveats
- LLM-based scores can be noisy; aggregate across seeds and prompts for stability
- CLIP scoring replaces `<phigep>` with a descriptive phrase to normalize prompts for CLIP
- Training scripts contain absolute paths; adjust for your environment

### License
No license file is provided. Ensure compliance with upstream licenses (e.g., FLUX.1-dev non-commercial license) and data usage policies when distributing code or models.