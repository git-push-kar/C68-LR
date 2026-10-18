# v3 Reasoning Repair — Stage-by-Stage Runbook

Prepared but **not executed**. The v2 approach is quarantined (see
`docs/quarantine/v2/`). v3 uses the *same* local base model and keeps the LoRA
adapter **external**.

- Base model (unmodified): `C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B`
- Windows venv: `.\venv\Scripts\python.exe` (torch 2.14 / transformers 4.47 — SFT, eval, chat)
- WSL2 venv: `~/vllm-env` (vLLM — teacher + rejection sampling **only**)
- v3 namespaces: `data/v3/`, `checkpoints/internvl3_2b_reasoning_v3/`, `models_v3/`

Decisions baked into the code: WSL2+vLLM · teacher `Qwen2.5-32B-Instruct-AWQ` ·
tag-based quarantine · whole-model wrap with LM-only targets · dependency-free
solver · adapter **and** merged+compare outputs.

---

## 0. One-time prep (no GPU)

### 0a. Free the GPU first
Every stage needs the whole A5000. Confirm it is idle before starting any run:
```powershell
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```
If it is not ~0, stop the holder (or reboot) and re-check.

### 0b. Quarantine v2 in git (do once)
```powershell
git tag approach-v2-quarantine
git add docs/ polyvalent_lr/ scripts/ tests/ data/v3 .gitignore README.md requirements.txt
git commit -m "v3: quarantine v2, add reasoning-repair pipeline (prepare only)"
git push origin master --tags
```
(The moves/copies of the v2 doc are already staged as files; the tag freezes the
last v2 commit.)

### 0c. Set up WSL2 + vLLM (teacher + sampling)
```bash
# inside WSL2 Ubuntu
bash /mnt/c/Users/Admin/Desktop/C68-LR/scripts/setup_wsl_vllm.sh
source ~/vllm-env/bin/activate
export HF_HOME=/mnt/d/hindifactguard/.cache/huggingface
```
> vLLM upgrades transformers — never install it into the Windows training venv
> (it would break InternVL's remote code path).

---

## Stage 0 — Recipe fix + masking check (~2–3 h incl. coding)

Masking is already implemented (`polyvalent_lr/reasoning_v3/data.py:encode_supervised`)
and the recipe lives in `polyvalent_lr/config_v3.py` (r64/α128, lr 1e-4, 3% warmup,
cosine, grad-clip 1.0, grad-ckpt on, max_len 1024).

```powershell
# 0.1 preview the teacher prompts (no model, safe on Windows)
.\venv\Scripts\python.exe scripts\gen_teacher_data.py --dry_run

# 0.2 verify assistant-only masking + EOS on a tiny sample (after Stage 1 exists)
.\venv\Scripts\python.exe scripts\inspect_batch.py --file data\v3\mix\train.jsonl --n 3

# 0.3 dry-run the SFT plan (no model load)
.\venv\Scripts\python.exe scripts\train_reasoning_v3.py --dry_run
```
**Smoke test (500 steps):** after the mixture exists, run SFT with
`--max_train_samples 16000` and confirm outputs stay fluent.
**Gate:** no gibberish; loss decreasing smoothly.

---

## Stage 1 — Dataset generation (~9–11 h, ~7 GPU)

### 1a. Set A — puzzles (CPU, Windows)
```powershell
.\venv\Scripts\python.exe scripts\gen_puzzles.py --n_total 20000 --holdout 300
```
Writes `data/v3/A/{train,holdout}.jsonl` (each example stores its seed so Stage 3
can regenerate + verify).

### 1b. Sets D & E — reused data (CPU, Windows)
```powershell
.\venv\Scripts\python.exe scripts\build_sft_mix.py --skip_prepare   # just checks dirs
```
(or let Stage 2's `build_sft_mix.py` materialise D/E automatically).

### 1c. Sets B & C — teacher (WSL2 + vLLM)
```bash
source ~/vllm-env/bin/activate
cd /mnt/c/Users/Admin/Desktop/C68-LR
python scripts/gen_teacher_data.py --set both --limit 6000 --holdout 300
```
Writes `data/v3/B/…` and `data/v3/C/…`. Check the teacher kept-rate in the log
(if very low, loosen the label matcher before scaling up).

---

## Stage 2 — SFT (~3.5–5 h)

```powershell
# assemble the mixture (also materialises D/E)
.\venv\Scripts\python.exe scripts\build_sft_mix.py
# train (rank 64, eval/save every 500 steps, keep best)
.\venv\Scripts\python.exe scripts\train_reasoning_v3.py `
  --output_dir checkpoints\internvl3_2b_reasoning_v3\sft
```
Adapter-only output at `checkpoints/internvl3_2b_reasoning_v3/sft/`.
**Benchmark the first 5 minutes** (tokens/s) and rescale the time estimate.
**Gate:** puzzle holdout ≥ ~40% on 4-entity puzzles; no Dolly chat collapse.

---

## Stage 3 — Rejection sampling / expert iteration (~2–4 h)

```bash
# sample + verify (WSL2 + vLLM)
source ~/vllm-env/bin/activate
python scripts/rejection_sampling.py `
  --stage2_adapter checkpoints/internvl3_2b_reasoning_v3/sft `
  --num_puzzles 8000 --samples 4 --temperature 0.7
# prints the exact retrain command, e.g.:
#   python scripts/train_reasoning_v3.py --init_adapter .../sft \
#     --train_file data/v3/rejection/accepted.jsonl --learning_rate 5e-5 \
#     --output_dir checkpoints/internvl3_2b_reasoning_v3/rs
```
Run the printed retrain command on Windows. Optionally repeat for a 2nd round.
**Gate:** ≥5 pts holdout gain; no Dolly / LogicBench regression.

---

## Stage 4 — Inference boosters + packaging

### 4a. Merged variant + adapter variant
```powershell
.\venv\Scripts\python.exe scripts\build_model_variants.py `
  --adapter_dir checkpoints\internvl3_2b_reasoning_v3\rs `
  --adapter_output models_v3\adapter --merged_output models_v3\merged
```

### 4b. Compare UI with self-consistency (temp 0.6, rep 1.1)
```powershell
.\venv\Scripts\python.exe scripts\chat_ui_compare.py `
  --model_a models_v3\adapter --adapter_a models_v3\adapter\adapter --label_a "v3 base+adapter" `
  --model_b models_v3\merged  --label_b "v3 merged" `
  --preset v3 --self_consistency 5 --port 8006
```
Open `http://127.0.0.1:8006`. In the UI set **Temperature 0.6** and
**Repetition penalty 1.1**. With `--self_consistency 5`, each answer samples 5
completions and majority-votes the final table (the vote is done server-side).

---

## Evaluation (throughout, ~1 GPU h)

```powershell
# base vs v3 on the assembled holdout (A/B/C/D/E)
.\venv\Scripts\python.exe scripts\eval_reasoning_v3.py --out evaluation_results_v3_base.json
.\venv\Scripts\python.exe scripts\eval_reasoning_v3.py `
  --adapter checkpoints\internvl3_2b_reasoning_v3\rs --self_consistency 5 `
  --out evaluation_results_v3_rs.json

# unseen benchmarks (existing harness)
.\venv\Scripts\python.exe scripts\evaluate_benchmarks.py `
  --adapter_dir checkpoints\internvl3_2b_reasoning_v3\rs --benchmark_dir data\benchmarks
```

---

## Time budget (benchmark & rescale)

| Stage | GPU h | Notes |
|---|---|---|
| 0 recipe + smoke | 0.5 | + ~2 h coding |
| 1 generation | ~7 | + ~3 h CPU/coding |
| 2 SFT | 3.5–5 | |
| 3 rejection round | 2–4 | |
| evals | ~1 | |
| **Total** | **~14–18** | ~1.5–2 days wall-clock with overnight runs |

## Go / no-go gates

| After | Gate |
|---|---|
| Stage 0 | No gibberish; loss smooth |
| Stage 2 | 4-entity puzzle holdout ≥ ~40%; inspect traces if lower |
| Stage 3 | ≥5-pt holdout gain; no Dolly-chat or LogicBench drop |

---

## Files added for v3

| Path | Purpose |
|---|---|
| `polyvalent_lr/config_v3.py` | v3 config (LoRA/SFT/rejection/decoding/teacher) |
| `polyvalent_lr/solvers/puzzle.py` | dependency-free puzzle generator + solver + validator |
| `polyvalent_lr/reasoning_v3/prompts.py` | clean `<reasoning>`/`<answer>` templates |
| `polyvalent_lr/reasoning_v3/data.py` | masking, dataset, mixture builder |
| `polyvalent_lr/reasoning_v3/decoding.py` | extraction + self-consistency majority vote |
| `scripts/gen_puzzles.py` | Set A (CPU) |
| `scripts/gen_teacher_data.py` | Sets B/C (WSL2 + vLLM) |
| `scripts/build_sft_mix.py` | assemble mixture + materialise D/E |
| `scripts/inspect_batch.py` | Stage-0 masking/EOS verification |
| `scripts/train_reasoning_v3.py` | Stage 2/3 SFT |
| `scripts/rejection_sampling.py` | Stage 3 sampling + verification |
| `scripts/eval_reasoning_v3.py` | holdout evaluation |
| `scripts/setup_wsl_vllm.sh` | one-time WSL2 + vLLM setup |
| `tests/test_puzzle_solver.py`, `tests/test_v3_masking.py`, `tests/test_v3_mix.py` | unit tests |

> **Important:** no stage has been run yet. Commands above are ready to execute
> once the GPU is free.
