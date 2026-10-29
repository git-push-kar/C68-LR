# v3 Benchmark Report

Model: `C:\Users\Admin\Desktop\LY_C68\deepfake_project\models\InternVL3-2B`

Adapter: `checkpoints\internvl3_2b_reasoning_v3\sft`

Adapter SHA-256: `f45cab3f76d76e918259428c874c62320e9d5322a84c21661b0f5d06fb4847a2`

Git commit: `355b08ee692dc38ed0d3f7373cbfc48e77d1635e`

GPU: NVIDIA RTX A5000 | torch 2.14.1+cu126 | transformers 4.47.1

Predictions: 3,200 | batch size 4 | sampling seed 42

Decoding: greedy (max_new_tokens=384) and SC5 (n=5, T=0.6, top_p=0.95, repetition_penalty=1.1).

Timing gate: 25 items in 122 s (4.87 s/item); projected full run 4.33 h.

## What this is

This is a zero-shot, inference-only evaluation of the released v3 adapter on three official public logic benchmarks plus the v3 in-distribution holdout. No benchmark data was added to training, no per-benchmark prompt tuning was done, and the base model was not evaluated by default. Accuracy is reported against the dataset's majority-class baseline and chance, not against an unstated reference model.

A benchmark is **headline-eligible** only if it passes the label-balance and prompt-uniqueness gates in `validate_data.py`. Multi-LogiEval fails the 70% label-balance gate (71.46% yes) and is therefore reported as a secondary diagnostic, not as a headline result.

## Provenance and gates

| Benchmark | Items | Evaluated | Max label share | Majority | Chance | Gate |
|---|---|---|---|---|---|---|
| LogicBench-BQA | 1520 | 600 | 65.8% | 65.8% | 50.0% | PASS |
| LogicBench-MCQA | 500 | 500 | 27.2% | 27.2% | 25.0% | PASS |
| LogicNLI | 2000 | 600 | 25.0% | 25.0% | 25.0% | PASS |
| Multi-LogiEval | 1696 | 600 | 71.5% | 71.5% | 50.0% | EXCLUDED |

Gate failures: {"Multi-LogiEval": ["majority label 'True' is 71.5% (> 70%)"]}.

Warnings: {"v3-Holdout/A": ["majority label 'unique' is 90.0%; diagnostic only (10% insufficient by design)"]}.

## Contamination audit

No benchmark item was derived from the v3 SFT corpora (A, D, E, or mix) by exact or near-duplicate content match.

| Benchmark | Items scanned | Contaminated | Exact matches | Any shared 8-gram | Max Jaccard |
|---|---|---|---|---|---|
| LogicBench-BQA | 1520 | 0 | 0 | 0 | 0.0 |
| LogicBench-MCQA | 500 | 0 | 0 | 0 | 0.0 |
| LogicNLI | 2000 | 0 | 0 | 0 | 0.0 |
| Multi-LogiEval | 1696 | 0 | 0 | 1 | 0.0051 |

This is an exact and 8-gram/near-duplicate audit against all four v3 training corpora (`A`, `D`, `E`, and the assembled `mix`). It is not a proof that no semantic overlap exists, and it does not audit the base model's pretraining corpus.

## All evaluated benchmarks

| Benchmark | Role | n | Accuracy [95% CI] | Majority | Chance | Malformed | Wrong (parsed) |
|---|---|---|---|---|---|---|---|
| LogicBench-BQA | headline | 600 | 41.5% [37.5%, 45.3%] | 65.3% | 50.0% | 0.0% | 58.5% |
| LogicBench-MCQA | headline | 500 | 73.0% [69.0%, 76.8%] | 27.2% | 25.0% | 0.0% | 27.0% |
| LogicNLI | headline | 600 | 27.2% [23.7%, 30.7%] | 25.0% | 25.0% | 0.0% | 72.8% |
| Multi-LogiEval | diagnostic | 600 | 51.8% [47.8%, 55.8%] | 72.0% | 50.0% | 0.0% | 48.2% |

`Malformed` means no valid label could be extracted; `Wrong (parsed)` means a valid label was extracted but did not match the gold label. They are deliberately separate.

## Headline results

| Benchmark | n | Accuracy [95% CI] | Majority | Chance | Malformed | Mean out tokens |
|---|---|---|---|---|---|---|
| LogicBench-BQA | 600 | 41.5% [37.5%, 45.3%] | 65.3% | 50.0% | 0.0% | 27.5 |
| LogicBench-MCQA | 500 | 73.0% [69.0%, 76.8%] | 27.2% | 25.0% | 0.0% | 30.9 |
| LogicNLI | 600 | 27.2% [23.7%, 30.7%] | 25.0% | 25.0% | 0.0% | 26.6 |

Majority/chance are reference points, not thresholds for success. A model at or below majority has not demonstrated benchmark-level reasoning on that set.

For the binary benchmarks, `Chance` is 50% because the gold task has two classes. The extractor also accepts the model's training label `Uncertain`, so a uniform three-way guesser would score 33.3%; the 50% column is the stricter binary-task reference.

Contamination sensitivity (all four v3 training corpora):

| Benchmark | Flagged | Accuracy with | Accuracy without |
|---|---|---|---|
| LogicBench-BQA | 0 | 41.5% | 41.5% |
| LogicBench-MCQA | 0 | 73.0% | 73.0% |
| LogicNLI | 0 | 27.2% | 27.2% |
| Multi-LogiEval | 0 | 51.8% | 51.8% |

When no item is flagged these columns are identical by construction; the flag file is still written per item.

### LogicBench-BQA

Per-label accuracy:

| Gold label | n | Accuracy [95% CI] | Malformed |
|---|---|---|---|
| True | 208 | 45.2% [38.5%, 51.9%] | 0 |
| False | 392 | 39.5% [34.7%, 44.4%] | 0 |

Confusion matrix (greedy):

| gold \ pred | True | False | Uncertain | <malformed> |
|---|---|---|---|---|
| True | 94 | 54 | 60 | 0 |
| False | 121 | 155 | 116 | 0 |
| Uncertain | 0 | 0 | 0 | 0 |

Most frequent prediction: `True` on 35.8% of items (gold share 34.7%).

Greedy vs SC5:

| Decoding | Accuracy [95% CI] | Malformed | Mean out tokens |
|---|---|---|---|
| greedy | 41.5% [37.5%, 45.3%] | 0 | 27.5 |
| SC5 | 34.7% [30.8%, 38.5%] | 0 | 27.7 |

Paired SC5 - greedy: -6.8% [95% CI -10.0%, -3.5%].

### LogicBench-MCQA

Per-label accuracy:

| Gold label | n | Accuracy [95% CI] | Malformed |
|---|---|---|---|
| choice_1 | 123 | 78.0% [70.7%, 85.4%] | 0 |
| choice_2 | 128 | 75.8% [68.0%, 82.8%] | 0 |
| choice_3 | 113 | 74.3% [66.4%, 82.3%] | 0 |
| choice_4 | 136 | 64.7% [56.6%, 72.8%] | 0 |

Confusion matrix (greedy):

| gold \ pred | choice_1 | choice_2 | choice_3 | choice_4 | choice_5 | <malformed> |
|---|---|---|---|---|---|---|
| choice_1 | 96 | 9 | 15 | 3 | 0 | 0 |
| choice_2 | 15 | 97 | 10 | 6 | 0 | 0 |
| choice_3 | 10 | 15 | 84 | 4 | 0 | 0 |
| choice_4 | 19 | 16 | 13 | 88 | 0 | 0 |
| choice_5 | 0 | 0 | 0 | 0 | 0 | 0 |

Most frequent prediction: `choice_1` on 28.0% of items (gold share 24.6%).

Greedy vs SC5:

| Decoding | Accuracy [95% CI] | Malformed | Mean out tokens |
|---|---|---|---|
| greedy | 73.0% [69.0%, 76.8%] | 0 | 30.9 |
| SC5 | 71.2% [67.2%, 75.0%] | 0 | 30.8 |

Paired SC5 - greedy: -1.8% [95% CI -4.2%, +0.6%].

### LogicNLI

Per-label accuracy:

| Gold label | n | Accuracy [95% CI] | Malformed |
|---|---|---|---|
| entailment | 150 | 10.0% [5.3%, 14.7%] | 0 |
| contradiction | 150 | 4.0% [1.3%, 7.3%] | 0 |
| neutral | 150 | 94.7% [90.7%, 98.0%] | 0 |
| self_contradiction | 150 | 0.0% [0.0%, 0.0%] | 0 |

Confusion matrix (greedy):

| gold \ pred | entailment | contradiction | neutral | self_contradiction | <malformed> |
|---|---|---|---|---|---|
| entailment | 15 | 0 | 135 | 0 | 0 |
| contradiction | 5 | 6 | 139 | 0 | 0 |
| neutral | 5 | 3 | 142 | 0 | 0 |
| self_contradiction | 19 | 6 | 125 | 0 | 0 |

Most frequent prediction: `neutral` on 90.2% of items (gold share 25.0%).

Greedy vs SC5:

| Decoding | Accuracy [95% CI] | Malformed | Mean out tokens |
|---|---|---|---|
| greedy | 27.2% [23.7%, 30.7%] | 0 | 26.6 |
| SC5 | 26.8% [23.3%, 30.3%] | 0 | 26.7 |

Paired SC5 - greedy: -0.3% [95% CI -1.2%, +0.5%].

Optional semantic mapping (`True -> entailment`, `False -> contradiction`, `Uncertain -> neutral`): greedy 27.2% [95% CI 23.7%, 30.7%], SC5 26.8% [95% CI 23.3%, 30.3%].

Strict-label accuracy above is primary. The mapped number is secondary and cannot recover `self_contradiction` because T/F/U has no corresponding token.

## Secondary diagnostic: Multi-LogiEval

Multi-LogiEval is **not** in the headline table because 71.46% of its labels are `True`, failing the pre-registered 70% label-balance gate. It is shown only to make the imbalance and the model's behavior visible.

| n | Accuracy [95% CI] | Majority | Chance | Malformed | Mean out tokens |
|---|---|---|---|---|---|
| 600 | 51.8% [47.8%, 55.8%] | 72.0% | 50.0% | 0.0% | 27.0 |

Per-depth accuracy:

| Depth | n | Accuracy [95% CI] | Majority |
|---|---|---|---|
| - | 49 | 49.0% [34.7%, 63.3%] | 85.7% |
| d1 | 156 | 57.7% [50.0%, 65.4%] | 92.9% |
| d2 | 153 | 47.1% [39.2%, 54.9%] | 52.9% |
| d3 | 104 | 54.8% [46.2%, 64.4%] | 64.4% |
| d4 | 95 | 48.4% [37.9%, 57.9%] | 69.5% |
| d5 | 43 | 51.2% [37.2%, 65.1%] | 93.0% |

The full per-file rule breakdown (111 file/rule groups) is in `summary.json` under `benchmarks.Multi-LogiEval.per_pattern`.

## Per-pattern accuracy (LogicBench)

### LogicBench-BQA

| Inference rule | n | Accuracy [95% CI] | Majority |
|---|---|---|---|
| bidirectional_dilemma | 63 | 49.2% [36.5%, 60.3%] | 74.6% |
| commutation | 31 | 29.0% [12.9%, 45.2%] | 74.2% |
| constructive_dilemma | 63 | 44.4% [31.7%, 55.6%] | 74.6% |
| default_reasoning_default | 16 | 37.5% [12.5%, 62.5%] | 50.0% |
| default_reasoning_irr | 16 | 0.0% [0.0%, 0.0%] | 50.0% |
| default_reasoning_open | 16 | 25.0% [6.2%, 50.0%] | 50.0% |
| default_reasoning_several | 31 | 16.1% [3.2%, 29.0%] | 74.2% |
| destructive_dilemma | 63 | 36.5% [23.8%, 49.2%] | 74.6% |
| disjunctive_syllogism | 32 | 0.0% [0.0%, 0.0%] | 50.0% |
| existential_generalization | 16 | 93.8% [81.2%, 100.0%] | 50.0% |
| hypothetical_syllogism | 63 | 52.4% [39.7%, 65.1%] | 74.6% |
| material_implication | 31 | 48.4% [32.3%, 67.7%] | 74.2% |
| modus_ponens | 16 | 93.8% [81.2%, 100.0%] | 50.0% |
| modus_tollens | 32 | 40.6% [25.0%, 59.4%] | 50.0% |
| reasoning_about_exceptions_1 | 31 | 41.9% [25.8%, 58.1%] | 74.2% |
| reasoning_about_exceptions_2 | 16 | 81.2% [62.5%, 100.0%] | 50.0% |
| reasoning_about_exceptions_3 | 16 | 18.8% [0.0%, 37.5%] | 50.0% |
| reasoning_about_priority | 32 | 34.4% [18.8%, 50.0%] | 50.0% |
| universal_instantiation | 16 | 75.0% [50.0%, 93.8%] | 50.0% |

### LogicBench-MCQA

| Inference rule | n | Accuracy [95% CI] | Majority |
|---|---|---|---|
| bidirectional_dilemma | 40 | 72.5% [57.5%, 85.0%] | 32.5% |
| commutation | 20 | 95.0% [85.0%, 100.0%] | 35.0% |
| constructive_dilemma | 40 | 90.0% [80.0%, 97.5%] | 32.5% |
| default_reasoning_default | 20 | 40.0% [20.0%, 60.0%] | 40.0% |
| default_reasoning_irr | 20 | 65.0% [45.0%, 85.0%] | 40.0% |
| default_reasoning_open | 20 | 20.0% [5.0%, 40.0%] | 45.0% |
| default_reasoning_several | 20 | 40.0% [20.0%, 60.0%] | 40.0% |
| destructive_dilemma | 40 | 92.5% [82.5%, 100.0%] | 30.0% |
| disjunctive_syllogism | 40 | 70.0% [55.0%, 83.8%] | 32.5% |
| existential_generalization | 20 | 100.0% [100.0%, 100.0%] | 40.0% |
| hypothetical_syllogism | 40 | 85.0% [75.0%, 95.0%] | 35.0% |
| material_implication | 20 | 85.0% [70.0%, 100.0%] | 50.0% |
| modus_ponens | 20 | 100.0% [100.0%, 100.0%] | 40.0% |
| modus_tollens | 40 | 77.5% [62.5%, 90.0%] | 30.0% |
| reasoning_about_exceptions_1 | 20 | 65.0% [45.0%, 85.0%] | 40.0% |
| reasoning_about_exceptions_2 | 20 | 60.0% [40.0%, 80.0%] | 45.0% |
| reasoning_about_exceptions_3 | 20 | 55.0% [35.0%, 75.0%] | 35.0% |
| reasoning_about_priority | 20 | 35.0% [15.0%, 55.0%] | 35.0% |
| universal_instantiation | 20 | 90.0% [75.0%, 100.0%] | 35.0% |

## v3 holdout (in-distribution sanity check)

These are not public-benchmark results. They are diagnostic checks on the v3 training distribution.

### A — logic puzzles

Exact-table accuracy (solver-validated). The stored targets for 30/300 `insufficient` puzzles are reference tables; scoring uses the regenerated puzzle, not the stored target.

| Subset | n | Greedy exact [95% CI] | SC5 exact [95% CI] |
|---|---|---|---|
| all | 300 | 14.7% [10.7%, 18.7%] | 15.7% [11.7%, 20.0%] |
| unique | 270 | 16.3% [11.9%, 20.7%] | 17.4% [13.0%, 22.2%] |
| insufficient | 30 | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] |

Majority baseline: always output the `unique` table = 90.0% exact (it is wrong on every insufficient item).

Per-cell accuracy on unique puzzles:

| Cell | Greedy | SC5 |
|---|---|---|
| person | 99.8% | 99.8% |
| domain | 87.0% | 87.2% |
| os | 87.2% | 87.5% |

Paired SC5 - greedy: +1.0% [95% CI -1.3%, +3.3%].

### D — open-ended explanations

| Metric | Greedy | SC5 |
|---|---|---|
| token-F1 [95% CI] | 36.1% [33.3%, 39.1%] | 31.9% [29.6%, 34.4%] |
| degenerate rate | 5.7% | 3.5% |
| truncated rate | 5.0% | 4.1% |
| tag-spam rate | 0.0% | 0.0% |
| mean repetition ratio | 0.160 | 0.147 |
| mean max repeated n-gram | 2.32 | 1.96 |

Mean greedy output: 96.0 tokens, 429 chars. Token-F1 is lexical overlap with the reference, not a correctness or quality score. The repetition ratio is `1 - unique_tokens/total_tokens`; the n-gram repeat count is the most frequent 2..8-gram count.

### E — label sanity

| n | Accuracy [95% CI] | Majority | Chance | Malformed |
|---|---|---|---|---|
| 300 | 83.7% [79.3%, 87.7%] | 39.0% | 33.3% | 0.0% |

Paired SC5 - greedy: +1.0% [95% CI -1.0%, +3.3%].

## General-chat degeneration probes

30 ordinary prompts, scored for emptiness, looping, truncation, and very short output. These are not a capability score.

| Failed | Degenerate (loop) | Loose n-gram flag | Empty | Truncated | <20 chars | Mean chars | Mean out tokens |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 3 | 0 | 0 | 0 | 544 | 115.8 |

`Degenerate (loop)` requires more than 60% of tokens to be repeats, so enumerated lists and recipes do not count. `Loose n-gram flag` shows how many outputs the generic 5x n-gram rule would flag; those were inspected and are structured answers, not loops.

Failure here means the adapter is not safe to present as a general assistant; it does not affect the logic-benchmark accuracy, which is reported separately.

## Plain-language conclusion

- **LogicBench-BQA**: 41.5% accuracy (95% CI 37.5%–45.3%) vs 65.3% majority and 50.0% chance; the point estimate is below the majority baseline.
- **LogicBench-MCQA**: 73.0% accuracy (95% CI 69.0%–76.8%) vs 27.2% majority and 25.0% chance; the point estimate is above the majority baseline.
- **LogicNLI**: 27.2% accuracy (95% CI 23.7%–30.7%) vs 25.0% majority and 25.0% chance; the point estimate is above the majority baseline.
- **LogicNLI label concentration**: `neutral` accounts for 90.2% of predictions, so the aggregate accuracy is dominated by one output label rather than balanced performance.
- **Multi-LogiEval** is excluded from the headline because its 71.46% yes rate fails the label-balance gate; see the diagnostic table.
- **SC5** is a paired decoding comparison to greedy on the same items; its CI is the uncertainty of that difference, not an independent effect claim.
- Only one model (the v3 adapter on the fixed base) was evaluated. No v2, base, or second system was run, so these are absolute accuracies; no comparative claim is made.

## Limitations

- **Single model, one checkpoint.** Only v3 (base + `internvl3_2b_reasoning_v3/sft`) is reported. `--include_base` is off by default, and no v2 result is comparable.
- **Sample sizes.** LogicBench-BQA uses a 600/1,520 stratified sample, Multi-LogiEval 600/1,696, LogicNLI 600/2,000, LogicBench-MCQA all 500, and each holdout split all 300. Bootstrap intervals reflect item-sampling variability only.
- **One seed.** Sampling uses seed 42 and bootstrap CIs use seed 20261028. Self-consistency is stochastic; these intervals do not include prompt, seed, or hyperparameter uncertainty.
- **Contamination scope.** The audit is exact normalized match plus 8-gram Jaccard/containment and MinHash confirmation over A, D, E, and mix. It is not a semantic-entailment audit, and it cannot rule out the base model having seen these public benchmarks during pretraining.
- **Label space.** The binary benchmarks keep their official True/False gold, but the model is allowed to emit `Uncertain` because that is its training contract; `Uncertain` is parsed and counted wrong, not malformed.
- Multi-LogiEval is label-imbalanced (71.5% True) and is not a headline result.
- LogicNLI uses the natural-language `test_language` split only; `test_logic` is symbolic and lacks a shipped verbaliser. Its four labels are a zero-shot output space not seen in v3 SFT.
- LogicBench-MCQA gold is the `choice_N` key; a model that writes the option text is mapped to a key by the extractor, but a model that writes only T/F/U is malformed. MCQA option keys are also a zero-shot output space.
- The v3 holdout is in-distribution and is not evidence of generalization. A has 30/300 `insufficient` items whose stored targets are reference tables; the scorer regenerates the puzzle and validates the answer kind.
- D has no gold labels, so token-F1 and degeneration are diagnostics, not accuracy. Token-F1 is lexical overlap, not correctness.
- **No human evaluation was performed.**
- **Outputs are short and templated.** Greedy answers average 27–31 tokens, and 88%–98% of them reduce to `Combining the premises, the answer is: X.`; these numbers measure final-label accuracy, not the quality of the reasoning trace.

## Artifacts

- `predictions.jsonl` — per-item greedy and SC5 outputs, labels, malformed flags, latencies.
- `summary.json` — machine-readable metrics, per-pattern breakdown, CIs.
- `timing.json` — 25-item gate and ETA.
- `run_config.json` — model/adapter hashes, versions, decoding, git state.
- `general_predictions.jsonl` / `general_summary.json` — general-chat probes.
