# Polyvalent: Logical Reasoning Integration

Project discussion note · 2 October 2026

## Scope and status

Polyvalent has three application domains: **Deepfake, Industrial Anomaly, and Finance RL**. Finance RL replaces the earlier Spatial domain. Logical Reasoning (LR) can either be a fourth independently routed capability or become part of the shared foundation supporting these applications.

**This note records a proposed training and experimental plan, not demonstrated results.** The discussion assumes existing application adapters can be retrained; their repository implementation and readiness were not independently verified. No measured LR gains, application improvements, or routing metrics are established here.

## LR training plan

Start with **InternVL3-2B and an LR LoRA**, using a staged curriculum rather than treating all datasets as equal-sized sources.

| Dataset | Proposed role | Purpose |
|---|---|---|
| ProofWriter | Main structured-deduction source | Learn inference from facts and rules, including multi-step deduction and proof structure. |
| P-FOLIO | Prioritize proof supervision | Learn from human-written reasoning traces and explicit proofs. |
| FOLIO | Smaller natural-language component | Broaden coverage through natural-language first-order reasoning and formal annotations. |
| AbductionRules | Optional small addition | Practice inferring explanations from observations; relevance to anomaly reasoning is a hypothesis to test. |

Proposed sequence: structured deduction with ProofWriter → richer proof and natural-language supervision with P-FOLIO/FOLIO → optionally add abduction. Sampling weights, curriculum boundaries, and hyperparameters remain experimental choices; no fixed percentages were agreed.

Normalize task labels and output formats while preserving distinctions between deduction and abduction. Split by underlying problem/premise group, check overlap between FOLIO and P-FOLIO, and prevent related examples or derived proofs from crossing training and validation boundaries.

Keep **LogicBench, Multi-LogiEval, and LogicNLI unseen for final evaluation**. Do not use their evaluation examples for training, curriculum selection, prompt tuning, or model selection. Use separate development data for those decisions. Compare the original base with base + LR LoRA under matched prompts and decoding, examining task accuracy and reasoning depth where available. Proof validity can be assessed where suitable annotations exist; fluent explanations alone do not establish correct reasoning.

## LR LoRA versus full base fine-tuning

| Approach | What training changes | Advantages | Limitations |
|---|---|---|---|
| LR LoRA | Small low-rank updates; original base frozen | Lower training/storage cost, easy rollback, isolated experiment, mergeable afterward | Restricted update capacity; improvement must be demonstrated |
| Full base fine-tuning on LR | All selected base parameters | Greater adaptation capacity | Higher resource cost; greater risk of forgetting or degrading other capabilities |

**Begin with LoRA.** Full base fine-tuning is an alternative experiment if LoRA proves insufficient. Merging a trained LoRA does not retroactively make its training equivalent to full fine-tuning.

## Architecture options

| Architecture | Structure | Pros | Cons / required checks |
|---|---|---|---|
| A: Four independent adapters | Original base + one selected Deepfake, Anomaly, Finance RL, or LR adapter | Modular, clean baseline, preserves original foundation and existing adapter compatibility | Application adapters do not automatically receive the LR update |
| B: Fully LR-fine-tuned foundation | Full LR base fine-tuning, then retrain three application adapters | Shared adaptation with greater parameter freedom | More expensive; check forgetting, multimodal capability, and downstream transfer |
| **C: Merged LR foundation** | Train LR LoRA, merge into a copy of the base, then retrain three application adapters | Shared LR update with efficient initial training; no separate LR adapter needed at deployment | Retraining required for a controlled comparison; LR gains and application transfer are uncertain |
| D: Adapter composition | Original base with LR and application adapters jointly active or combined | Potentially reusable LR component; flexible per-domain combinations | Composition support and method must be explicit; independently trained updates can interfere and gains are not additive |

**Architecture C is the preferred shared-foundation experiment**, given the assumption that retraining all three application adapters is affordable. Architecture A remains the simplest baseline and fallback. Architecture D should be tested separately rather than assumed equivalent to C.

## Architecture C and the meaning of “merge/fuse”

For each targeted weight matrix, standard LoRA learns:

```text
W ∈ R^(d_out × d_in)
A ∈ R^(r × d_in), B ∈ R^(d_out × r)
ΔW_LR = s · B A          (typically s = α/r)
W_effective = W + ΔW_LR
```

Before merging, the base and adapter are stored separately. **Merging adds the scaled learned update to the corresponding base weight matrices**, then saves a standalone checkpoint:

```text
Original InternVL3-2B + trained LR LoRA
                    ↓ merge into a copy
              InternVL3-2B-LR
                    ↓ freeze as new foundation
          ┌─────────┼──────────┐
          ↓         ↓          ↓
      Deepfake    Industrial   Finance RL
      new LoRA   Anomaly LoRA   new LoRA
```

The original checkpoint and LR adapter should be retained. The merged checkpoint incorporates the LR update and does not require the separate LR adapter for inference. For standard additive LoRA, merged and unmerged inference should agree up to numerical precision; verify this in the actual implementation, especially if quantization is involved.

Each application adapter is **trained anew against the LR-enhanced foundation**. Its effective targeted weights become:

```text
W_domain = W_LR + ΔW_domain,new
         = W + ΔW_LR + ΔW_domain,new
```

Simply attaching an old application adapter trained against the original base is a different experiment. Likewise, loading two independently trained adapters together is composition, not the sequential training procedure above. The presence of LR updates in the foundation does not guarantee retained reasoning or better application performance.

## Hybrid routing

Use the following priority order:

1. **Explicit metadata:** validated task/domain selection from the API, UI, or caller takes precedence; a Finance simulator already identifies its task.
2. **Deterministic modality/format checks:** recognize task-specific schemas and supported inputs. Modality narrows candidates but does not uniquely identify a domain.
3. **Learned domain router:** classify ambiguous requests using task intent and input features, with confidence thresholds.
4. **General/LR fallback:** use the general or LR-capable path for low-confidence requests; seek clarification when specialized execution requires an unambiguous domain.

An image can belong to Deepfake or Industrial Anomaly; numerical/time-series inputs can belong to Anomaly or Finance RL. Avoid image-only or text-only domain rules. Under Architecture A the fallback can select the LR adapter; under C it uses the LR-enhanced base without an application adapter. Measure routing accuracy and end-to-end outcomes separately.

## Minimum experiment plan

1. Establish original-base and original-base-plus-application-adapter baselines.
2. Train LR LoRA and evaluate on the reserved unseen logic benchmarks.
3. If LR improvement is supported, merge into a copy and verify inference equivalence and general/multimodal regressions.
4. Retrain Deepfake, Industrial Anomaly, and Finance RL adapters on that foundation; compare with matched original-base training using the same data and comparable budgets.
5. Recheck LR after application adaptation. Report per-domain gains and regressions, with appropriate repeated-run uncertainty; optional composition and full-fine-tuning experiments can follow.

**Hypotheses:** LR training improves unseen logical reasoning; a shared LR foundation helps one or more application domains; specialization retains useful LR capability. Each requires evidence. Stronger text-logic scores alone do not establish better deepfake detection, anomaly detection, or Finance RL performance.

## Constraint/symbolic verifier

A constraint or symbolic verifier is **optional, not necessary for the core architecture**. LR adapter training plus credible unseen evaluation is sufficient to test the LR contribution. A verifier would add formalization/parsing, solver integration, coverage limits, and new failure modes.

Consider it only after the LR and application experiments are stable, as a separate ablation measuring whether it improves validity and task outcomes. Do not make it a prerequisite or imply that LoRA training itself guarantees formally valid reasoning.

## Working decision

Build the independent LR LoRA first. Keep Architecture A as the modular baseline, then test Architecture C by merging LR into a copy of InternVL3-2B and retraining all three application adapters. Use hybrid routing and reserve symbolic verification for a later extension. Adopt the shared foundation only if evaluation supports its benefits.
