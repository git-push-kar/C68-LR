"""Unit tests for the advanced benchmark answer extractor."""
import pytest
from polyvalent_lr.evaluation.bench_extract import parse_prediction, extract_answer_span


def test_parser_extract_answer_tag():
    raw = "Let us think.\n<reasoning>Step 1</reasoning>\n<answer>True</answer>"
    item = {"label_space": ["True", "False", "Uncertain"]}
    parsed = parse_prediction(raw, item)
    assert parsed["label"] == "True"
    assert parsed["malformed"] is False
    assert parsed["answer_span_method"] == "answer_tag"


def test_parser_extract_answer_line():
    raw = "We examine the rules.\nTherefore, Final Answer: False"
    item = {"label_space": ["True", "False", "Uncertain"]}
    parsed = parse_prediction(raw, item)
    assert parsed["label"] == "False"
    assert parsed["malformed"] is False
    assert parsed["answer_span_method"] == "answer_line"


def test_parser_extract_last_line():
    raw = "Based on the premises,\nUncertain."
    item = {"label_space": ["True", "False", "Uncertain"]}
    parsed = parse_prediction(raw, item)
    assert parsed["label"] == "Uncertain"
    assert parsed["malformed"] is False


def test_parser_mcqa_extraction():
    raw = "Option 2 is logically entailed. <answer>choice_2</answer>"
    item = {"label_space": ["choice_1", "choice_2", "choice_3", "choice_4"]}
    parsed = parse_prediction(raw, item)
    assert parsed["label"] == "choice_2"
    assert parsed["malformed"] is False


def test_parser_nli_semantic_aliases():
    raw = "The hypothesis contradicts the facts. <answer>contradiction</answer>"
    item = {"label_space": ["entailment", "contradiction", "neutral", "self_contradiction"]}
    parsed = parse_prediction(raw, item)
    assert parsed["label"] == "contradiction"
    assert parsed["malformed"] is False


def test_parser_malformed_empty():
    raw = "No valid conclusion can be made at all and no tag exists."
    item = {"label_space": ["choice_1", "choice_2", "choice_3", "choice_4"]}
    parsed = parse_prediction(raw, item)
    assert parsed["label"] is None
    assert parsed["malformed"] is True
