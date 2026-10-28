from polyvalent_lr.evaluation.bench_extract import (
    canonicalize,
    extract_answer_span,
    parse_prediction,
)

BINARY = {
    "label_space": ["True", "False"],
    "aliases": {"yes": "True", "no": "False", "true": "True", "false": "False"},
    "choices": None,
}
BINARY3 = {
    "label_space": ["True", "False", "Uncertain"],
    "aliases": {
        "yes": "True", "no": "False", "true": "True", "false": "False",
        "uncertain": "Uncertain", "unknown": "Uncertain",
        "undetermined": "Uncertain",
    },
    "choices": None,
}
NLI = {
    "label_space": ["entailment", "contradiction", "neutral", "self_contradiction"],
    "aliases": {
        "entailment": "entailment",
        "contradiction": "contradiction",
        "neutral": "neutral",
        "self-contradiction": "self_contradiction",
    },
    "choices": None,
}
MCQA = {
    "label_space": ["choice_1", "choice_2", "choice_3", "choice_4"],
    "aliases": {
        "choice_1": "choice_1", "1": "choice_1", "a": "choice_1",
        "choice_2": "choice_2", "2": "choice_2", "b": "choice_2",
        "choice_3": "choice_3", "3": "choice_3", "c": "choice_3",
        "choice_4": "choice_4", "4": "choice_4", "d": "choice_4",
    },
    "choices": {
        "choice_1": "If Jane consumes water, she gets a sugar crash",
        "choice_2": "John will feel hydrated",
        "choice_3": "She will feel hydrated or she does not eat too much sugar",
        "choice_4": "Jane will not feel hydrated",
    },
}


def test_answer_tag():
    rec = parse_prediction("<reasoning>x</reasoning><answer>True</answer>", BINARY)
    assert rec["label"] == "True"
    assert rec["answer_span_method"] == "answer_tag"
    assert rec["malformed"] is False


def test_answer_line():
    rec = parse_prediction("reasoning...\nanswer: False", BINARY)
    assert rec["label"] == "False"
    assert rec["answer_span_method"] == "answer_line"


def test_last_line_fallback():
    rec = parse_prediction("reasoning...\ntherefore yes", BINARY)
    assert rec["label"] == "True"
    assert rec["answer_span_method"] == "last_line"


def test_last_valid_label_token_wins():
    label, reason = canonicalize("first True then False", BINARY["label_space"], BINARY["aliases"])
    assert label == "False"
    assert reason == "token"


def test_repeated_tag_spam():
    text = "</answer></answer></answer><answer><answer>True</answer></answer>"
    rec = parse_prediction(text, BINARY)
    assert rec["label"] == "True"


def test_empty_answer_tag_falls_through():
    rec = parse_prediction("<answer>   </answer>\nFalse", BINARY)
    assert rec["label"] == "False"


def test_uncertain_is_not_binary():
    rec = parse_prediction("<answer>Uncertain</answer>", BINARY)
    assert rec["label"] is None
    assert rec["malformed"] is True
    assert rec["match_reason"] == "no_label"


def test_holdout_three_way_uncertain():
    rec = parse_prediction("<answer>Uncertain</answer>", BINARY3)
    assert rec["label"] == "Uncertain"
    assert rec["malformed"] is False


def test_binary_three_way_unknown_alias_is_parsed():
    for answer in ("unknown", "undetermined", "Uncertain"):
        rec = parse_prediction(f"<answer>{answer}</answer>", BINARY3)
        assert rec["label"] == "Uncertain"
        assert rec["malformed"] is False


def test_binary_three_way_uncertain_is_wrong_not_malformed():
    # Gold is binary; the extractor still accepts the model's training label,
    # so the outcome is a parsed wrong answer rather than an unparsable one.
    rec = parse_prediction("<answer>Uncertain</answer>", BINARY3)
    assert rec["label"] != "True" and rec["label"] != "False"
    assert rec["malformed"] is False


def test_nli_strict_and_semantic():
    strict = parse_prediction("<answer>entailment</answer>", NLI)
    assert strict["label"] == "entailment"
    wrong_space = parse_prediction("<answer>True</answer>", NLI)
    assert wrong_space["label"] is None
    mapped = parse_prediction("<answer>True</answer>", NLI, allow_semantic_nli=True)
    assert mapped["label"] == "entailment"
    assert mapped["match_reason"] == "semantic"


def test_nli_self_contradiction_aliases():
    for answer in ("self_contradiction", "self-contradiction", "self contradiction"):
        rec = parse_prediction(f"<answer>{answer}</answer>", NLI)
        assert rec["label"] == "self_contradiction"


def test_mcqa_key_number_letter_and_text():
    for answer in ("choice_3", "3", "c", "C"):
        rec = parse_prediction(f"<answer>{answer}</answer>", MCQA)
        assert rec["label"] == "choice_3", answer
    text = "<answer>She will feel hydrated or she does not eat too much sugar</answer>"
    rec = parse_prediction(text, MCQA)
    assert rec["label"] == "choice_3"
    assert rec["match_reason"] in {"exact", "option_text"}


def test_mcqa_wrong_space_binary_is_malformed():
    rec = parse_prediction("<answer>True</answer>", MCQA)
    assert rec["label"] is None


def test_empty_generation():
    rec = parse_prediction("", BINARY)
    assert rec["label"] is None
    assert rec["answer_span_method"] == "empty"


def test_extract_answer_span_ignores_reasoning_tags():
    span, method = extract_answer_span("<reasoning><answer>False</answer></reasoning>")
    assert span == "False"
    assert method == "answer_tag"
