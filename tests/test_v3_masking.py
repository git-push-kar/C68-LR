"""Tests for v3 assistant-only masking (prompt -> -100, target + EOS kept)."""
from polyvalent_lr.reasoning_v3.data import encode_supervised


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(ord(c) % 90) + 1 for c in text]}


def test_prompt_masked_and_eos_supervised():
    tok = FakeTokenizer()
    enc = encode_supervised(tok, "AB", "CD", max_length=50)
    assert enc["input_ids"] == [ord("A") % 90 + 1, ord("B") % 90 + 1,
                                ord("C") % 90 + 1, ord("D") % 90 + 1, 0]
    assert enc["labels"][:2] == [-100, -100]
    assert enc["labels"][-1] == tok.eos_token_id
    assert enc["labels"][2:] == enc["input_ids"][2:]
    assert enc["attention_mask"] == [1] * 5


def test_target_eos_not_doubled():
    tok = FakeTokenizer()

    class TokWithEos(FakeTokenizer):
        def __call__(self, text, add_special_tokens=False):
            ids = [(ord(c) % 90) + 1 for c in text]
            if text == "CD":
                ids = ids + [0]
            return {"input_ids": ids}

    enc = encode_supervised(TokWithEos(), "AB", "CD", max_length=50)
    assert enc["labels"][-1] == 0
    assert enc["labels"].count(0) == 1


def test_prompt_truncation_preserves_target():
    tok = FakeTokenizer()
    enc = encode_supervised(tok, "ABCDEFGH", "XY", max_length=4)
    supervised = [t for t in enc["labels"] if t != -100]
    assert supervised[-1] == 0
    assert len(enc["input_ids"]) <= 4
