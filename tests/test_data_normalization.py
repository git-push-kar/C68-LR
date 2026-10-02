import pytest
from polyvalent_lr.data.normalizer import DatasetNormalizer
from polyvalent_lr.data.schema import LogicalExample, LogicTaskType
from polyvalent_lr.config import DatasetSource


def test_label_normalization():
    assert DatasetNormalizer.normalize_label("entailment") == "True"
    assert DatasetNormalizer.normalize_label("proved") == "True"
    assert DatasetNormalizer.normalize_label("contradiction") == "False"
    assert DatasetNormalizer.normalize_label("disproved") == "False"
    assert DatasetNormalizer.normalize_label("neutral") == "Uncertain"
    assert DatasetNormalizer.normalize_label("unknown") == "Uncertain"
    assert DatasetNormalizer.normalize_label(True) == "True"
    assert DatasetNormalizer.normalize_label(False) == "False"


def test_proofwriter_normalization():
    raw_item = {
        "id": "pw_test_1",
        "theory_id": "theory_10",
        "theory": "If someone is smart, they read books. Alice is smart.",
        "question": "Alice reads books.",
        "answer": "True",
        "proof": "Alice is smart -> Alice reads books"
    }
    example = DatasetNormalizer.normalize_proofwriter(raw_item)
    assert example.id == "pw_pw_test_1"
    assert example.group_id == "pw_group_theory_10"
    assert example.source == DatasetSource.PROOFWRITER
    assert example.task_type == LogicTaskType.DEDUCTION
    assert len(example.premises) == 2
    assert example.label == "True"
    assert example.proof_steps is not None
    assert len(example.proof_steps) == 1


def test_instruction_formatting():
    example = LogicalExample(
        id="test_ex",
        group_id="grp_1",
        source=DatasetSource.FOLIO,
        task_type=LogicTaskType.DEDUCTION,
        premises=["All birds fly.", "Penguins are birds."],
        hypothesis_or_query="Penguins fly.",
        label="True",
        proof_trace_text="Universal rule applied to penguins."
    )
    prompt = example.to_instruction_prompt()
    target = example.to_target_text()

    assert "Premises:" in prompt
    assert "- All birds fly." in prompt
    assert "Hypothesis:\nPenguins fly." in prompt
    assert "<think>\nUniversal rule applied to penguins.\n</think>" in target
    assert "<answer>True</answer>" in target
