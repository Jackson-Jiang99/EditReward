# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

EditReward is a VLM-based reward model for scoring instruction-guided image edits (ICLR 2026). It is a single Python package (`EditReward/`) — not a monorepo, no web server, no database, no Docker. All workflows (inference, training, evaluation) are CLI-driven with Python/DeepSpeed.

### Environment

- **Python 3.10** is required (matches the project's conda instructions). A venv at `/workspace/.venv` is pre-created.
- **PyTorch 2.5.1** is installed in CPU mode (no GPU on this VM). On a GPU machine, install with `--index-url https://download.pytorch.org/whl/cu124` instead.
- **Flash Attention** is optional; the code falls back to SDPA automatically. The "Flash Attention is not installed" warning is benign.
- The `rich` package is an unlisted dependency of `trl==0.8.6` — it is included in the update script.

### Activating the environment

```bash
source /workspace/.venv/bin/activate
```

### Key commands

| Task | Command |
|---|---|
| Lint (critical errors) | `flake8 EditReward/ --select=E9,F63,F7 --max-line-length=150` |
| Verify imports | `python -c "from EditReward import EditRewardInferencer; print('OK')"` |
| Parse a config | `python -c "from EditReward.utils.parser import *; from omegaconf import OmegaConf; print(OmegaConf.load('EditReward/config/EditReward-Qwen2.5-7B-VL.yaml'))"` |
| Training (GPU required) | `deepspeed EditReward/train_qwen_vl_edit.py --config EditReward/config/EditReward-Qwen2.5-7B-VL.yaml` |
| Inference (GPU required) | See `EditReward/infer_edit.py` |
| Evaluation (GPU or API) | See `EditReward/evaluate/README.md` |

### Gotchas

- **bf16 config**: The YAML configs set `bf16: True`, which causes `TrainingArguments` to fail on CPU. When testing config parsing on CPU, load the YAML via `OmegaConf.load()` directly (or remove `deepspeed`/`bf16` keys before passing to `HfArgumentParser`).
- **No `requirements.txt` / `pyproject.toml`**: Dependencies are documented only in README.md inline pip commands. The update script below replicates them.
- **Model checkpoints**: Full inference requires downloading model checkpoints from HuggingFace (`TIGER-Lab/EditReward-*`). These are multi-GB files not included in the repo.
- **`model/test_differentiable.py`** is a manual test script (not pytest), and it requires GPU + a checkpoint to run.
- **Two F821 flake8 warnings** in `model/differentiable_image_processor.py` are false positives (string-based `"PIL.Image.Image"` type annotations).
