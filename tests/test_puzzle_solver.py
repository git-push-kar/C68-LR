"""Tests for the dependency-free puzzle generator/solver (Set A + Stage 3)."""
from polyvalent_lr.solvers import puzzle as pz


def test_generated_puzzles_are_unique_and_valid():
    for seed in range(15):
        p = pz.generate_puzzle(seed, n=4)
        assert p is not None
        sols = pz.solve(p.people, p.domain_values, p.os_values, p.clues, p.n, limit=2)
        assert len(sols) == 1
        assert sols[0] == p.solution
        assert pz.validate_answer(p, p.render_table())
        assert pz.validate_answer(p, p.render_trace())


def test_wrong_answer_is_rejected():
    p = pz.generate_puzzle(7, n=4)
    assert p is not None
    wrong = p.render_table().replace(str(p.solution[1]["person"]), "Nobody", 1)
    assert not pz.validate_answer(p, wrong)


def test_insufficient_puzzle_requires_not_enough_information():
    found = None
    for seed in range(200):
        p = pz.generate_puzzle(seed, n=4, insufficient=True)
        if p is not None:
            found = p
            break
    assert found is not None, "no insufficient puzzle generated"
    assert found.answer_kind == "insufficient"
    assert pz.validate_answer(found, "Not enough information to determine the layout.")
    assert not pz.validate_answer(found, found.render_table())


def test_check_clue_partial_semantics():
    clue = pz.Clue("fixed_person", {"desk": 2, "person": "Alice"}, "Alice sits at desk 2.")
    assert pz.check_clue(clue, {}, 3) is None
    assert pz.check_clue(clue, {2: {"person": "Alice", "domain": "AI", "os": "Linux"}}, 3) is True
    assert pz.check_clue(clue, {3: {"person": "Alice", "domain": "AI", "os": "Linux"}}, 3) is False
