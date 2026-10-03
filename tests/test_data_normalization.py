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
    # Official ProofWriter nested format
    raw_item = {
        "id": "RelNeg-OWA-D2-1717",
        "maxD": 2,
        "theory": "The cow is big. If something is big then it chases the dog.",
        "triples": {
            "triple1": {"text": "The cow is big.", "representation": "(\"cow\" \"is\" \"big\" \"+\")"}
        },
        "rules": {
            "rule3": {"text": "If something is big then it chases the dog.", "representation": "..."}
        },
        "questions": {
            "Q1": {
                "question": "The cow is big.",
                "answer": True,
                "QDep": 0,
                "strategy": "proof",
                "proofsWithIntermediates": [{"representation": "triple1", "intermediates": []}]
            },
            "Q7": {
                "question": "The dog does not chase the dog.",
                "answer": "Unknown",
                "QDep": 1,
                "strategy": "inv-rconc"
            }
        }
    }
    examples = DatasetNormalizer.normalize_proofwriter(raw_item)
    assert len(examples) == 2
    
    # Check Q1
    q1_ex = next(e for e in examples if e.metadata["question_id"] == "Q1")
    assert q1_ex.id == "pw_RelNeg-OWA-D2-1717_Q1"
    assert q1_ex.group_id == "pw_theory_RelNeg-OWA-D2-1717"
    assert q1_ex.label == "True"
    assert "The cow is big." in q1_ex.premises
    assert "If something is big then it chases the dog." in q1_ex.premises
    
    # Check Q7
    q7_ex = next(e for e in examples if e.metadata["question_id"] == "Q7")
    assert q7_ex.label == "Uncertain"
    assert q7_ex.group_id == "pw_theory_RelNeg-OWA-D2-1717"



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
