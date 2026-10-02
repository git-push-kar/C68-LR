import pytest
from polyvalent_lr.evaluation.metrics import ReasoningMetrics


def test_metrics_extraction():
    sample_output = (
        "<think>\n"
        "1. Alex is a student who studies.\n"
        "2. Universal rule implies Alex passes the exam.\n"
        "</think>\n"
        "<answer>True</answer>"
    )
    res = ReasoningMetrics.evaluate_prediction(sample_output, ground_truth_label="True")
    assert res["exact_match"] is True
    assert res["pred_label"] == "True"
    assert res["has_reasoning_trace"] is True
    assert res["reasoning_depth"] == 2


def test_aggregate_metrics():
    eval_results = [
        {"exact_match": True, "has_reasoning_trace": True, "reasoning_depth": 3},
        {"exact_match": True, "has_reasoning_trace": True, "reasoning_depth": 2},
        {"exact_match": False, "has_reasoning_trace": True, "reasoning_depth": 1},
        {"exact_match": False, "has_reasoning_trace": False, "reasoning_depth": 0},
    ]
    agg = ReasoningMetrics.compute_aggregate_metrics(eval_results)
    assert agg["accuracy"] == 50.0
    assert agg["reasoning_trace_rate"] == 75.0
    assert agg["avg_reasoning_depth"] == 1.5
    assert agg["total_examples"] == 4
