"""Stage 0 verification: decode a few supervised examples and show the labels.

Confirms that (a) only assistant tokens are supervised, and (b) the EOS
(``<|im_end|>``) is included in the labels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvalent_lr import config_v3
from polyvalent_lr.reasoning_v3.data import encode_supervised, read_jsonl

try:  # the Windows console defaults to cp1252 and chokes on unicode traces
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _safe(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect assistant-only masking.")
    ap.add_argument("--file", type=str, default=str(config_v3.MIX_DIR / "train.jsonl"))
    ap.add_argument("--model", type=str, default=config_v3.BASE_MODEL_PATH)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=config_v3.SFTV3().max_length)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    examples = read_jsonl(Path(args.file))
    if not examples:
        raise SystemExit(f"no examples in {args.file}")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    print(f"eos_token_id={eos_id} pad_token_id={getattr(tokenizer, 'pad_token_id', None)}")

    unmasked_leak = 0
    for i, ex in enumerate(examples[: args.n]):
        enc = encode_supervised(tokenizer, ex.prompt, ex.target, args.max_length)
        ids, labels = enc["input_ids"], enc["labels"]
        n_prompt = sum(1 for l in labels if l == -100)
        supervised = [t for t in labels if t != -100]
        print("=" * 80)
        print(f"[{i}] category={ex.category} task={ex.task}  (prompt {n_prompt} tok masked, "
              f"{len(supervised)} supervised)")
        print(f"    prompt   : {_safe(tokenizer.decode(ids[:n_prompt], skip_special_tokens=False)[:400])!r}")
        print(f"    supervised: {_safe(tokenizer.decode(supervised, skip_special_tokens=False)[:400])!r}")
        print(f"    ends_with_eos: {bool(supervised) and supervised[-1] == eos_id}")

    for ex in examples:
        enc = encode_supervised(tokenizer, ex.prompt, ex.target, args.max_length)
        supervised = [t for t in enc["labels"] if t != -100]
        if not supervised or supervised[-1] != eos_id:
            unmasked_leak += 1
    print("=" * 80)
    print(f"examples missing EOS in labels: {unmasked_leak}/{len(examples)}")


if __name__ == "__main__":
    main()
