# QUARANTINE — v2 approach (do not reuse)

This folder freezes the **v2** logical-reasoning approach. It is retained for
history/audit only. **Nothing here should be reused, imported, or overwritten**
by the v3 pipeline.

## Why it was quarantined

The v2 adapter (base + LoRA, rank 16/α32, curriculum ProofWriter→P-FOLIO/FOLIO→
AbductionRules) did **not** fix the base model's degeneration:

- raw scaffold leakage (`reasoning … <br /> … answer …`, `<answered>`, stray
  HTML/`<pre>`/`<code>`/`<sup>` tags),
- unsolved grid/ordering puzzles (invents desks, treats a straight row as
  circular, contradictory tables),
- arithmetic errors (`9.9` vs `9.11`),
- fragile name tracking (fabricates names),
- partially preserved but inconsistent general chat.

See `CURRENT_APPROACH.md` inside this folder for the full write-up and the raw
side-by-side transcript that motivated the quarantine.

## What is frozen

- `docs/quarantine/v2/CURRENT_APPROACH.md` — the v2 method + evidence.
- Git tag **`approach-v2-quarantine`** — points at the last v2 commit.
- v2 artifacts (gitignored, left in place, treated as read-only):
  - `checkpoints/internvl3_2b_lr_v2/` (and `internvl3_2b_lr/`, `internvl3_2b_lr_fused/`)
  - `models/lr_adapter/`, `models/lr_merged/`
  - `data/processed/` is **not** quarantined — sets D (Dolly) and E
    (ProofWriter/FOLIO labels) are legitimately reused by v3.

## Where v3 lives

The new approach is entirely separate: see `docs/v3/RUNBOOK.md`. v3 writes only
to `data/v3/`, `checkpoints/internvl3_2b_reasoning_v3/`, and `models_v3/`.
