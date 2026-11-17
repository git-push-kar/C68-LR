"""Unit tests for perturbation transformations."""
import pytest
from scripts.perturb import (
    rename_entities,
    shuffle_premises,
    add_distractors,
    paraphrase_templates,
    negate_conclusion,
    build_perturbed_item,
)


def test_rename_entities():
    premises = ["Charlie is red.", "Alice is smart.", "The dog chased the cat."]
    question = "Is Charlie red?"
    new_p, new_q, app = rename_entities(premises, question)
    assert "Xylor-7 is red." in new_p[0]
    assert "Zenth-4 is smart." in new_p[1]
    assert "flump-B chased the greeble-A." in new_p[2]
    assert "Is Xylor-7 red?" == new_q
    assert "Charlie" in app
    assert "Alice" in app


def test_shuffle_premises():
    premises = ["Premise A.", "Premise B.", "Premise C.", "Premise D."]
    s1 = shuffle_premises(premises, seed=10)
    s2 = shuffle_premises(premises, seed=20)
    assert set(s1) == set(premises)
    assert set(s2) == set(premises)
    assert len(s1) == len(premises)


def test_add_distractors():
    premises = ["Fido is a dog.", "Dogs are mammals."]
    d1 = add_distractors(premises, n_distractors=1, seed=42)
    assert len(d1) == 3
    assert set(premises).issubset(set(d1))

    d3 = add_distractors(premises, n_distractors=3, seed=42)
    assert len(d3) == 5
    assert set(premises).issubset(set(d3))


def test_paraphrase_templates():
    premises = [
        "If Charlie is red, then Charlie is big.",
        "All birds are animals.",
        "Fido is happy.",
    ]
    para = paraphrase_templates(premises)
    assert "Whenever Charlie is red, it is guaranteed that Charlie is big." in para[0]
    assert "Every member of the group of birds is classified as animals." in para[1]
    assert "Fido exhibits the quality of being happy." in para[2]


def test_negate_conclusion():
    q1, g1, v1 = negate_conclusion("Charlie is red.", "True")
    assert q1 == "Charlie is not red."
    assert g1 == "False"
    assert v1 is True

    q2, g2, v2 = negate_conclusion("Anne is not cold.", "False")
    assert q2 == "Anne is cold."
    assert g2 == "True"
    assert v2 is True

    q3, g3, v3 = negate_conclusion("Dave is big.", "Uncertain")
    assert q3 == "Dave is not big."
    assert g3 == "Uncertain"
    assert v3 is True


def test_build_perturbed_item():
    item = {
        "id": "item_123",
        "premises": ["Bob is green."],
        "question": "Is Bob green?",
        "gold": "True",
        "label_space": ["True", "False", "Uncertain"],
    }
    p_item = build_perturbed_item(item, "rename_entities")
    assert p_item is not None
    assert "Quux-9" in p_item["premises"][0]
    assert p_item["id"].startswith("item_123__pert_")
