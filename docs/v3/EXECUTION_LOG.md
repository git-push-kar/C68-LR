# v3 Reasoning-Repair — Execution Log

Date: 2026-10-19 → 2026-10-20
Host: Windows 10/11 workstation, NVIDIA RTX A5000 (24 GB), non-admin shell
Repo: `C:\Users\Admin\Desktop\C68-LR` (branch `master`)
Interpreter: `.\venv\Scripts\python.exe`
AnyDesk: **left running throughout** (4 processes before and after; never stopped).

This file records every stage that was executed, the exact command, wall-clock
time, key terminal output, and the metrics produced. Raw logs are in
`docs/v3/_logs/`.

---

## 1. Executive summary

| Stage | Status | Result |
|-------|--------|--------|
| 0. Environment / dry runs | DONE | teacher preview + puzzle solver verified; 9/9 tests pass |
| 1a. Set A puzzles | DONE | 20,000 train + 300 holdout |
| 1b. Set B (ProofWriter teacher) | **BLOCKED** | vLLM/WSL2 unavailable (see §5) |
| 1c. Set C (FOLIO/Abduction teacher) | **BLOCKED** | same vLLM/WSL2 blocker |
| 1d. Set D rehearsal | DONE | 13,913 train + 300 holdout |
| 1e. Set E labels | DONE | 514,769 train + 300 holdout (cleaned) |
| 1f. SFT mixture | DONE (partial) | A/D/E only → 72,153 train, ~21.87 M tokens/epoch |
| 2. LoRA SFT | DONE | 20,701 s (5 h 45 m), train_loss 3.029, eval_loss 0.739 |
| 3. Rejection sampling | **BLOCKED** | vLLM-only script; dry run OK (12,065 × 4 samples planned) |
| 4. Evaluation | DONE (sampled) | A 6 % strict, E 92 %, D metric artifact 0 % |
| 5. Packaging + UI | DONE | adapter + merged verified; HTTP UI serves both |

Headline: the v3 adapter learns the contract and the label task well
(E = 92 %), but still makes **one domain error per puzzle** on strict full-table
puzzles (A = 6 %). See §9 for the diagnosis and the next experiment.

---

## 2. Environment snapshot

```
GPU           : NVIDIA RTX A5000, 24564 MiB, ~730 MiB used (idle)
Python        : .\venv\Scripts\python.exe
torch         : 2.14.1+cu126   transformers: 4.47.1   peft: 0.21.2
bitsandbytes  : 0.50.2         accelerate: 1.15.0      datasets: 5.0.1
vllm / trl / z3: NOT installed
WSL           : distro "Ubuntu" registered, **Stopped / unusable**
Admin         : False
Virtualization firmware: True
AnyDesk       : running (PIDs 9836, 14492, 17600, 30024)
```

WSL diagnostics (verbatim):

```
$ wsl --status
Default Distribution: Ubuntu
Default Version: 2
WSL1 is not supported with your current machine configuration.
Please enable the "Windows Subsystem for Linux" optional component to use WSL1.
WSL2 is not supported with your current machine configuration.
Please enable the "Virtual Machine Platform" optional component and ensure
virtualization is enabled in the BIOS.
Enable "Virtual Machine Platform" by running: wsl.exe --install --no-distribution

$ wsl -e echo hi
The operation could not be started because a required feature is not installed.
Error code: Wsl/Service/CreateInstance/CreateVm/HCS/HCS_E_SERVICE_NOT_AVAILABLE
```

Remediation requires **admin + reboot**: enable *Windows Subsystem for Linux* and
*Virtual Machine Platform*, then `wsl --install --no-distribution`. Not possible
in this session.

---

## 3. Stage 0 — smoke tests

```powershell
.\venv\Scripts\python.exe scripts\gen_teacher_data.py --dry_run
```
```
[dry_run] wrote 25 prompt previews to data\v3\B\_prompt_preview.txt
EXIT_CODE=0 DURATION_SECONDS=14.37
```

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_puzzle_solver.py tests\test_v3_masking.py tests\test_v3_mix.py -q
```
```
9 passed in 2.35s
```

### 3.1 Solver performance (bug found and fixed)

The first `gen_puzzles.py --n_total 50 --holdout 5` **timed out after 30 min**.
Profiling showed the uniqueness proof exploded for larger puzzles:

| entities n | before fix | after fix |
|-----------:|-----------:|----------:|
| 3 | 0.007 s | ~0.001 s |
| 4 | 0.151 s | 0.001 s |
| 5 | 15.58 s | 0.003 s |
| 6 | (not finishable in practice) | 0.006 s |

Fix in `polyvalent_lr/solvers/puzzle.py` (`_select_minimal_unique`):
seed **two absolute anchor clues** before relational clues so the uniqueness
proof starts from a pinned desk; dropped the O(k²) redundant-clue pruning pass.
The generated puzzles remain unique and are still validated by `solve()`.

---

## 4. Stage 1 — data

### 4.1 Set A (solver-generated puzzles)

First full generation was fast but produced **targets too long**:

```
prompt  p50/p90/p99/max = 290/352/365/371
target  p50/p90/p99/max = 830/1529/1641/1696
target > 1024 tokens: 8056/20000
```

Cause: `render_trace()` repeated every clue text and a **full snapshot of all
desks per clue**, and `render_trace()` embedded the final table which
`build_puzzle_target()` appended a second time in `<answer>`.

Fix: compact trace that only reports desks that become fully determined, no
repeated clue list, table only in `<answer>`. Also reserve the EOS when a target
is truncated (`reasoning_v3/data.py`).

Final generation:

```powershell
.\venv\Scripts\python.exe scripts\gen_puzzles.py --n_total 20000 --holdout 300
```
```
[Set A] wrote 20000 train + 300 holdout -> data\v3\A
EXIT_CODE=0 DURATION_SECONDS=1087.04      # 18.1 min
```

Final target stats: `p50 149 / p90 273 / p99 326 / max 338`; prompt max 366;
**0 targets over 700 tokens**.

### 4.2 Sets D and E

```powershell
.\venv\Scripts\python.exe scripts\build_sft_mix.py
```
```
[prepare D] wrote 13913 train + 300 holdout
[prepare E] wrote 514769 train + 300 holdout
```

Set E was cleaned before the final build:
- dropped dataset debug strings (`[@0: ... [CWA ... (FAIL)]`);
- replaced them with `Combining the premises, the answer is: <label>.`;
- switched E from the puzzle system prompt to `SYSTEM_LOGIC`
  (no “update each desk” instruction).

### 4.3 Mixture (bug found and fixed)

First build over-allocated E because the sampler did not track the remaining
token budget:

```
[mix] wrote 485074 train + 900 holdout (~150,528,644 tokens/epoch)   # WRONG
```

Fixed with a budget-preserving, weight-renormalising allocation (missing B/C
must not under-fill the mix). Final:

```
[mix] weights = {'A': 0.45, 'B': 0.2, 'C': 0.1, 'D': 0.1, 'E': 0.15}
[mix] per-set counts: {'A': 20000, 'B': 0, 'C': 0, 'D': 13913, 'E': 38758}
[mix] wrote 72153 train + 900 holdout (~21,868,746 tokens/epoch)
EXIT_CODE=0 DURATION_SECONDS=35.89
```

`inspect_batch.py` after the fixes:

```
[0] category=D task=chat  (prompt 51 tok masked, 10 supervised)
[1] category=E task=labels (prompt 219 tok masked, 28 supervised)
[2] category=E task=labels (prompt 157 tok masked, 28 supervised)
examples missing EOS in labels: 0/72153
```

---

## 5. Stage 1b/1c and Stage 3 — BLOCKED (vLLM on Windows)

`vllm` has no native Windows build; the plan was WSL2 Ubuntu. WSL2 cannot start
on this machine because the *Virtual Machine Platform* feature is not enabled
and enabling it needs admin + reboot (see §2). Therefore:

- Set B (ProofWriter teacher rationales) — not generated.
- Set C (FOLIO/Abduction teacher rationales) — not generated.
- Stage 3 rejection sampling — not run.

The Stage 3 dry run is valid and confirms the prompt set is ready:

```powershell
.\venv\Scripts\python.exe scripts\rejection_sampling.py --dry_run
```
```
[RS] 12065 puzzles x 4 samples @ T=0.7
[dry_run] wrote preview -> data\v3\rejection\_rs_prompt_preview.txt
EXIT_CODE=0
```

The real command fails at `from vllm import LLM` until WSL2 is available.

Consequence: the mixture uses **A + D + E only**, with weights renormalised:
A 64.3 %, D 14.3 %, E 21.4 % of the token budget.

---

## 6. Stage 2 — LoRA SFT

Plan (`--dry_run`):

```
train / eval    : 72,153 / 900
batch/accum/eff : 4/8/32
epochs / steps  : 2 / ~4508
lr / warmup     : 0.0001 / 0.03
eval/save every : 500 steps
```

Smoke run (`--max_train_samples 512 --epochs 1`):

```
16/16 [01:10<00:00, 4.20s/it]
{'train_runtime': 71.0953, 'train_steps_per_second': 0.225, 'train_loss': 5.8605}
EXIT_CODE=0 DURATION_SECONDS=89.27
```

Full run (detached process, PID 1760):

```powershell
.\venv\Scripts\python.exe scripts\train_reasoning_v3.py
```

```
[LoRA Setup] Trainable params: 73,859,072 || All params: 2,162,816,512 (3.41%)
{'eval_loss': 0.7618, 'epoch': 0.22}
{'eval_loss': 0.7415, 'epoch': 0.44}
{'eval_loss': 0.7294, 'epoch': 0.67}
{'eval_loss': 0.7094, 'epoch': 0.89}
{'eval_loss': 0.7522, 'epoch': 1.11}
{'eval_loss': 0.7324, 'epoch': 1.33}
{'eval_loss': 0.7354, 'epoch': 1.55}
{'eval_loss': 0.7291, 'epoch': 1.77}
{'eval_loss': 0.7395, 'epoch': 2.00}
{'train_runtime': 20701.1551, 'train_samples_per_second': 6.971,
 'train_steps_per_second': 0.218, 'train_loss': 3.0289, 'epoch': 2.0}
[v3] adapter saved to checkpoints\internvl3_2b_reasoning_v3\sft
```

Wall clock: **20,701 s = 5 h 45 m 1 s** (~4.59 s/step). No CUDA/OOM errors.
Artifact: `checkpoints/internvl3_2b_reasoning_v3/sft` (~296 MB adapter).

> Note: the eval loss is dominated by the short E targets, so it is a
> convergence check, not the capability metric. See §7.

---

## 7. Stage 4 — Evaluation

Evaluation uses plain `transformers` (no vLLM) and the solver validator for Set A.

### 7.1 Smoke (12 A puzzles, T=0.6)

```
"A": {"correct": 1, "total": 12, "accuracy": 0.0833}
DURATION_SECONDS=217.87
```

### 7.2 Adapter, greedy, 150 balanced examples (50 A / 50 D / 50 E)

```
"A": {"correct": 3,  "total": 50, "accuracy": 0.06}
"D": {"correct": 0,  "total": 50, "accuracy": 0.0}
"E": {"correct": 46, "total": 50, "accuracy": 0.92}
DURATION_SECONDS=1751.34      # 29.2 min
```

### 7.3 Self-consistency (5 samples, T=0.6, 20 A puzzles)

```
"A": {"correct": 2, "total": 20, "accuracy": 0.10}
DURATION_SECONDS=1799.32      # 30.0 min
```

Self-consistency did not rescue the domain errors (within noise on n=20).

### 7.4 Failure analysis (manual probes)

- **Set A**: names, desks and OS are usually right; the error is almost always
  **one domain per puzzle** (e.g. gold `Elena/Graphics` → predicted
  `Elena/Security`; gold `Noah/QA` → predicted `Noah/Networking`). Set A’s exact
  match requires the whole 4-column table, so near-misses score 0.
- **Set D**: generations are correct paraphrases
  (e.g. Rayleigh-scattering answer vs. the short gold sentence). The D metric
  uses substring matching, so it reports 0 % — a **metric artifact**, not a
  capability regression.
- **Set E**: 92 % confirms the `<reasoning>/<answer>` contract and label task.

Failed attempt worth recording: the first 150-example eval was run with buffered
stdout; it was killed by the tool timeout at 90 min and wrote no log/JSON. The
successful rerun used `python -u` and `--max_new_tokens 384`.

---

## 8. Stage 5 — Packaging and UI

### 8.1 Build variants

```powershell
.\venv\Scripts\python.exe scripts\build_model_variants.py `
  --base_model 'C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B' `
  --adapter_dir 'checkpoints\internvl3_2b_reasoning_v3\sft' `
  --adapter_output 'models_v3\adapter' --merged_output 'models_v3\merged'
```
```
[Equivalence Check] Max diff: 1.812500 | Mean diff: 0.157227 | Passed: True
[Variants] Equivalence check PASSED (base+adapter == merged within tolerance).
[Variants] Done.
EXIT_CODE=0 DURATION_SECONDS=35.03
```

Base signature unchanged. Layout:
`models_v3/adapter/` (base copy + `adapter/` subdir) and `models_v3/merged/`.

### 8.2 UI bug fixed

The compare UI’s v3 preset still injected the v2 ` thinking` format, so the
first self-test produced ` thinking ...` (and the merged model produced
malformed `</thinks>`/`< answer >`). Fix in `scripts/chat_ui_compare.py`:
v3 now uses `V3_SYSTEM`, never appends `LOGIC_FORMAT`, stops at `</answer>`,
and the self-test uses the v3 prompt.

Second self-test:

```powershell
.\venv\Scripts\python.exe scripts\chat_ui_compare.py `
  --model_a models_v3\adapter --model_b models_v3\merged `
  --label_a v3-adapter --label_b v3-merged --preset v3 --self_test
```
```
--- v3 adapter ---
<reasoning>
Tom is a mammalian.
</reasoning>

<answer>
True
</answer>

--- v3 merged ---
<reasoning>
Tom is a mammalian.
</reasoning>

<answer>
True
</answer>
DURATION_SECONDS=27.33
```

### 8.3 HTTP surface

```
GET /            -> 200, 9,762 bytes
GET /api/info    -> models: v3-adapter, v3-merged
POST /api/chat   -> 200, SSE stream for a and b, both:
  <reasoning> Tom is a mammalian. </reasoning>
  <answer> True </answer>
```

Server was stopped by PID after the test; AnyDesk remained alive.

---

## 9. Diagnosis and next experiment

The adapter learned the output contract and the label task, but not the
**domain-assignment** part of the puzzle. The compact trace only shows desks
that become *fully* determined, so the model never sees how a domain is pinned
down incrementally; it guesses domains consistent with the person/desk/OS
pattern.

Highest-value next change (requires one more 5–6 h SFT run):

1. Regenerate Set A with a **medium-detail trace**: after each clue, emit newly
   fixed *attributes* (`person`, `domain`, `os`) individually, not only complete
   desks. Target length should stay ≤ ~500 tokens.
2. Add a small **domain-focused curriculum**: filter/oversample puzzles where
   the only remaining ambiguity is a domain.
3. Re-run Stage 2 (same recipe) and compare A strict accuracy on the same
   150-example balanced holdout.
4. Unblock Stage 3 by enabling WSL2 as admin:
   `wsl --install --no-distribution`, enable *Virtual Machine Platform*,
   reboot; then `scripts/setup_wsl_vllm.sh` and the real
   `rejection_sampling.py`.

Secondary: make the D metric paraphrase-tolerant (token F1 / embedding) so it
stops reporting 0 % for correct answers.

---

## 10. Artifacts

| Path | Contents |
|------|----------|
| `data/v3/A/{train,holdout}.jsonl` | 20,000 + 300 compact-trace puzzles |
| `data/v3/D/{train,holdout}.jsonl` | 13,913 + 300 Dolly rehearsal |
| `data/v3/E/{train,holdout}.jsonl` | 514,769 + 300 cleaned label examples |
| `data/v3/mix/{train,holdout}.jsonl` | 72,153 + 900 (A/D/E only) |
| `data/v3/mix/holdout_balanced_150.jsonl` | 50 A / 50 D / 50 E eval subset |
| `checkpoints/internvl3_2b_reasoning_v3/sft` | final LoRA adapter |
| `models_v3/adapter` | base copy + `adapter/` subdir |
| `models_v3/merged` | fused standalone model |
| `docs/v3/_logs/` | raw logs for every command above |

All commands above were run from `C:\Users\Admin\Desktop\C68-LR` with
`.\venv\Scripts\python.exe`. No base-model files were modified.
