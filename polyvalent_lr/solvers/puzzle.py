"""Dependency-free generator + solver for grid/ordering ("zebra") puzzles.

Used by the v3 reasoning pipeline for two things:

* **Set A** — generating training examples (trace + final table), CPU only.
* **Stage 3** — verifying a model's final answer during rejection sampling.

A puzzle places ``N`` people in ``N`` desks (a straight row, 1..N). Each person
has a unique domain and a unique operating system, so the unknowns are the full
mapping ``desk -> (person, domain, os)``. Clues are English statements that are
all true in the reference solution.

The generator keeps only clue sets whose solution is *unique* (and equal to the
reference). For the ~10% "insufficient information" cases it keeps a clue set
with **no** unique solution and a query whose value genuinely varies across
solutions; the correct answer there is "Not enough information".

Stdlib only — no ``z3`` or other third-party dependency (decision Q6: option 1).
"""
from __future__ import annotations

import itertools
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- selector categories -------------------------------------------------------
PERSON = "person"
DOMAIN = "domain"
OS = "os"
CATEGORIES = (PERSON, DOMAIN, OS)

Selector = Tuple[str, str]  # (category, value)

# --- vocabularies --------------------------------------------------------------
PERSON_NAMES: List[str] = [
    "Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Liam", "Mona", "Noah", "Olivia", "Peter",
    "Quinn", "Rosa", "Sam", "Tara", "Umar", "Vera", "Will", "Xena",
    "Yara", "Zane", "Aarav", "Bianca", "Carlos", "Diya", "Elena", "Farid",
]
DOMAIN_VALUES: List[str] = [
    "AI", "Security", "Cloud", "Frontend", "Backend", "Data", "DevOps",
    "Embedded", "Graphics", "Networking", "Systems", "QA",
]
OS_VALUES: List[str] = [
    "Linux", "macOS", "Windows", "FreeBSD", "OpenBSD", "Android", "iOS",
    "Ubuntu", "Fedora", "Solaris", "ChromeOS", "Debian",
]


class BudgetExceeded(Exception):
    """Raised when the search exceeds its node budget (near-degenerate clue set)."""


# ==============================================================================
# Data structures
# ==============================================================================
@dataclass
class Clue:
    kind: str
    params: Dict[str, Any]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "params": self.params, "text": self.text}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Clue":
        return Clue(kind=d["kind"], params=d["params"], text=d["text"])


@dataclass
class Puzzle:
    n: int
    people: List[str]
    domain_values: List[str]
    os_values: List[str]
    solution: Dict[int, Dict[str, str]]  # desk -> {person, domain, os}
    clues: List[Clue]
    answer_kind: str                      # "unique" | "insufficient"
    query: Optional[str] = None           # for insufficient: what was asked
    seed: int = 0

    # -- rendering --------------------------------------------------------------
    def clue_lines(self) -> List[str]:
        return [f"{i}. {c.text}" for i, c in enumerate(self.clues, start=1)]

    def render_problem(self) -> str:
        body = "\n".join(self.clue_lines())
        return (
            f"{self.n} engineers ({', '.join(self.people)}) sit in a straight row of "
            f"{self.n} desks, numbered 1 to {self.n} from left to right. Each engineer "
            f"works in a different domain and uses a different primary OS.\n\n"
            f"Determine the exact desk number, domain, and OS for every engineer based "
            f"on these rules:\n\n{body}\n\n"
            f"Provide your thinking step-by-step, updating what you know about each desk "
            f"after evaluating each clue. Conclude with a clean markdown table showing the "
            f"final layout."
        )

    def render_table(self) -> str:
        rows = [
            "| Desk | Name | Domain | OS |",
            "|------|------|--------|----|",
        ]
        for d in range(1, self.n + 1):
            a = self.solution[d]
            rows.append(f"| {d} | {a['person']} | {a['domain']} | {a['os']} |")
        return "\n".join(rows)

    def render_trace(self) -> str:
        """Compact derivation: only report desks that become fully determined.

        The clue list already lives in the prompt, so it is not repeated here and
        the final table is emitted by the ``<answer>`` block only. This keeps the
        supervised target well inside the context budget.
        """
        lines: List[str] = []
        prev: Dict[int, Optional[tuple]] = {d: None for d in range(1, self.n + 1)}
        for k in range(1, len(self.clues) + 1):
            known = known_per_desk(self, self.clues[:k], cap=500)
            for d in range(1, self.n + 1):
                vals = tuple(known[d][cat] for cat in CATEGORIES)
                if all(v is not None for v in vals) and vals != prev[d]:
                    prev[d] = vals
                    person, domain, os_ = vals
                    lines.append(f"Clue {k} fixes desk {d}: {person}, {domain}, {os_}.")
        if not lines:
            lines.append("Together the clues determine the layout uniquely.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "people": self.people,
            "domain_values": self.domain_values,
            "os_values": self.os_values,
            "solution": {str(k): v for k, v in self.solution.items()},
            "clues": [c.to_dict() for c in self.clues],
            "answer_kind": self.answer_kind,
            "query": self.query,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Puzzle":
        return Puzzle(
            n=d["n"],
            people=list(d["people"]),
            domain_values=list(d["domain_values"]),
            os_values=list(d["os_values"]),
            solution={int(k): dict(v) for k, v in d["solution"].items()},
            clues=[Clue.from_dict(c) for c in d["clues"]],
            answer_kind=d["answer_kind"],
            query=d.get("query"),
            seed=d.get("seed", 0),
        )


# ==============================================================================
# Solver
# ==============================================================================
def _position_of(sel: Selector, assignment: Dict[int, Dict[str, str]]) -> Optional[int]:
    cat, val = sel
    for d, a in assignment.items():
        if a.get(cat) == val:
            return d
    return None


def check_clue(clue: Clue, assignment: Dict[int, Dict[str, str]], n: int) -> Optional[bool]:
    """Return True (satisfied), False (violated) or None (undetermined yet)."""
    k, p = clue.kind, clue.params

    if k == "fixed_person":
        if p["desk"] in assignment:
            return assignment[p["desk"]]["person"] == p["person"]
        pos = _position_of((PERSON, p["person"]), assignment)
        return False if (pos is not None and pos != p["desk"]) else None

    if k == "fixed_domain":
        if p["desk"] in assignment:
            return assignment[p["desk"]]["domain"] == p["domain"]
        pos = _position_of((DOMAIN, p["domain"]), assignment)
        return False if (pos is not None and pos != p["desk"]) else None

    if k == "fixed_os":
        if p["desk"] in assignment:
            return assignment[p["desk"]]["os"] == p["os"]
        pos = _position_of((OS, p["os"]), assignment)
        return False if (pos is not None and pos != p["desk"]) else None

    if k == "person_domain":
        pos = _position_of((PERSON, p["person"]), assignment)
        if pos is None:
            return None
        val = assignment[pos].get("domain")
        return None if val is None else (val == p["domain"])

    if k == "person_os":
        pos = _position_of((PERSON, p["person"]), assignment)
        if pos is None:
            return None
        val = assignment[pos].get("os")
        return None if val is None else (val == p["os"])

    if k == "domain_os":
        pos = _position_of((DOMAIN, p["domain"]), assignment)
        if pos is None:
            return None
        val = assignment[pos].get("os")
        return None if val is None else (val == p["os"])

    if k == "not_desk_os":
        if p["desk"] in assignment:
            return assignment[p["desk"]]["os"] != p["os"]
        return None

    if k == "left_of":
        pa = _position_of((p["a"]["cat"], p["a"]["val"]), assignment)
        pb = _position_of((p["b"]["cat"], p["b"]["val"]), assignment)
        if pa is not None and pb is not None:
            return pa < pb
        return None

    if k == "next_to":
        pa = _position_of((p["a"]["cat"], p["a"]["val"]), assignment)
        pb = _position_of((p["b"]["cat"], p["b"]["val"]), assignment)
        if pa is not None and pb is not None:
            return abs(pa - pb) == 1
        return None

    if k == "between":
        pa = _position_of((p["a"]["cat"], p["a"]["val"]), assignment)
        pm = _position_of((p["mid"]["cat"], p["mid"]["val"]), assignment)
        pc = _position_of((p["c"]["cat"], p["c"]["val"]), assignment)
        if None not in (pa, pm, pc):
            return min(pa, pc) < pm < max(pa, pc)
        return None

    raise ValueError(f"unknown clue kind: {k}")


def solve(
    people: Sequence[str],
    domain_values: Sequence[str],
    os_values: Sequence[str],
    clues: Sequence[Clue],
    n: int,
    limit: int = 2,
    node_budget: int = 1_500_000,
) -> List[Dict[int, Dict[str, str]]]:
    """Enumerate up to ``limit`` solutions consistent with ``clues``."""
    if n != len(people) or n != len(domain_values) or n != len(os_values):
        raise ValueError("pool sizes must equal n")
    if limit <= 1:
        raise ValueError("limit must be > 1 to distinguish unique from non-unique")

    results: List[Dict[int, Dict[str, str]]] = []
    assignment: Dict[int, Dict[str, str]] = {}
    used = {PERSON: set(), DOMAIN: set(), OS: set()}
    nodes = [0]

    def backtrack(desk: int) -> None:
        if len(results) >= limit:
            return
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise BudgetExceeded(f"exceeded {node_budget} nodes")
        if desk > n:
            for c in clues:
                if check_clue(c, assignment, n) is not True:
                    return
            results.append({d: dict(a) for d, a in assignment.items()})
            return

        for person in people:
            if person in used[PERSON]:
                continue
            for dom in domain_values:
                if dom in used[DOMAIN]:
                    continue
                for o in os_values:
                    if o in used[OS]:
                        continue
                    assignment[desk] = {"person": person, "domain": dom, "os": o}
                    used[PERSON].add(person)
                    used[DOMAIN].add(dom)
                    used[OS].add(o)
                    consistent = True
                    for c in clues:
                        if check_clue(c, assignment, n) is False:
                            consistent = False
                            break
                    if consistent:
                        backtrack(desk + 1)
                    used[PERSON].discard(person)
                    used[DOMAIN].discard(dom)
                    used[OS].discard(o)
                    del assignment[desk]
                    if len(results) >= limit:
                        return

    backtrack(1)
    return results


def known_per_desk(puzzle: Puzzle, clues: Sequence[Clue], cap: int = 200) -> Dict[int, Dict[str, Optional[str]]]:
    """For each desk, the attributes fixed across *all* consistent solutions."""
    sols = solve(
        puzzle.people, puzzle.domain_values, puzzle.os_values, clues, puzzle.n, limit=cap
    )
    known: Dict[int, Dict[str, Optional[str]]] = {}
    for d in range(1, puzzle.n + 1):
        entry: Dict[str, Optional[str]] = {}
        for cat in CATEGORIES:
            vals = {s.get(d, {}).get(cat) for s in sols}
            entry[cat] = vals.pop() if len(vals) == 1 else None
        known[d] = entry
    return known


def _snapshot_text(known: Dict[int, Dict[str, Optional[str]]], n: int) -> str:
    parts: List[str] = []
    for d in range(1, n + 1):
        bits = [f"{cat}={known[d][cat]}" for cat in CATEGORIES if known[d][cat]]
        parts.append(f"desk{d}[" + (", ".join(bits) if bits else "?") + "]")
    return " ".join(parts)


# ==============================================================================
# Clue generation
# ==============================================================================
def _sellabel(sel: Dict[str, str]) -> str:
    cat, val = sel["cat"], sel["val"]
    if cat == PERSON:
        return val
    if cat == DOMAIN:
        return f"the {val} engineer"
    return f"the engineer who uses {val}"


def _all_true_clues(solution: Dict[int, Dict[str, str]], n: int, rng: random.Random) -> List[Clue]:
    desks = list(range(1, n + 1))
    clues: List[Clue] = []

    for d in desks:
        a = solution[d]
        clues.append(Clue("fixed_person", {"desk": d, "person": a["person"]},
                           f"{a['person']} sits at desk {d}."))
        clues.append(Clue("fixed_domain", {"desk": d, "domain": a["domain"]},
                           f"The {a['domain']} engineer sits at desk {d}."))
        clues.append(Clue("fixed_os", {"desk": d, "os": a["os"]},
                           f"The engineer at desk {d} uses {a['os']}."))
        clues.append(Clue("person_domain", {"person": a["person"], "domain": a["domain"]},
                           f"{a['person']} is the {a['domain']} engineer."))
        clues.append(Clue("person_os", {"person": a["person"], "os": a["os"]},
                           f"{a['person']} uses {a['os']}."))
        clues.append(Clue("domain_os", {"domain": a["domain"], "os": a["os"]},
                           f"The {a['domain']} engineer uses {a['os']}."))

    # relational / ordering clues
    for _ in range(3 * n):
        i, j = sorted(rng.sample(desks, 2))
        cat = rng.choice(CATEGORIES)
        a = {"cat": cat, "val": solution[i][cat]}
        b = {"cat": cat, "val": solution[j][cat]}
        clues.append(Clue("left_of", {"a": a, "b": b},
                          f"{_sellabel(a)} sits somewhere to the left of {_sellabel(b)}."))

    for _ in range(n):
        i, j = sorted(rng.sample(desks, 2))
        if j - i != 1:
            continue
        cat = rng.choice(CATEGORIES)
        a = {"cat": cat, "val": solution[i][cat]}
        b = {"cat": cat, "val": solution[j][cat]}
        clues.append(Clue("next_to", {"a": a, "b": b},
                          f"{_sellabel(a)} sits exactly next to {_sellabel(b)}."))

    for _ in range(n):
        triple = sorted(rng.sample(desks, 3))
        if len(triple) < 3:
            continue
        a, mid, c = triple
        cat = rng.choice(CATEGORIES)
        sel_a = {"cat": cat, "val": solution[a][cat]}
        sel_m = {"cat": cat, "val": solution[mid][cat]}
        sel_c = {"cat": cat, "val": solution[c][cat]}
        clues.append(Clue("between", {"a": sel_a, "mid": sel_m, "c": sel_c},
                          f"{_sellabel(sel_m)} sits exactly between {_sellabel(sel_a)} "
                          f"and {_sellabel(sel_c)}."))

    # a few negative constraints
    for _ in range(n):
        d = rng.choice(desks)
        wrong = [o for o in OS_VALUES if o != solution[d]["os"]]
        o = rng.choice(wrong)
        clues.append(Clue("not_desk_os", {"desk": d, "os": o},
                          f"The engineer at desk {d} does not use {o}."))

    rng.shuffle(clues)
    return clues


def _is_unique(
    people, domains, oses, clues, n, solution, budget
) -> bool:
    sols = solve(people, domains, oses, clues, n, limit=2, node_budget=budget)
    return len(sols) == 1 and _same_solution(sols[0], solution)


def _same_solution(a: Dict[int, Dict[str, str]], b: Dict[int, Dict[str, str]]) -> bool:
    return all(a[d] == b[d] for d in a)


def _select_minimal_unique(
    candidates: List[Clue], people, domains, oses, n, solution, rng, budget
) -> Optional[List[Clue]]:
    """Build a unique clue set quickly.

    Two absolute "anchor" clues are seeded first: they pin a couple of desks and
    make the uniqueness proof cheap (without anchors the search over the three
    permutations can explode). Remaining clues are added one at a time until the
    solution is unique. Redundancy pruning is intentionally skipped — a cheap,
    varied clue set is more valuable here than a minimal one.
    """
    fixed = [c for c in candidates if c.kind.startswith("fixed_")]
    links = [c for c in candidates if c.kind in {"person_domain", "person_os", "domain_os"}]
    rel = [c for c in candidates if c.kind in {"left_of", "next_to", "between", "not_desk_os"}]
    for lst in (fixed, links, rel):
        rng.shuffle(lst)

    n_anchor = 2 if n >= 4 else 1
    selected: List[Clue] = list(fixed[:n_anchor])

    try:
        if selected and _is_unique(people, domains, oses, selected, n, solution, budget):
            return selected
        for c in fixed[n_anchor:] + links + rel:
            selected.append(c)
            if _is_unique(people, domains, oses, selected, n, solution, budget):
                return selected
    except BudgetExceeded:
        return None
    return None


def generate_puzzle(
    seed: int,
    n: Optional[int] = None,
    min_entities: int = 3,
    max_entities: int = 6,
    insufficient: bool = False,
    budget: int = 1_500_000,
    max_attempts: int = 60,
) -> Optional[Puzzle]:
    """Generate one puzzle (unique by default, else an 'insufficient' one)."""
    rng = random.Random(seed)
    for _ in range(max_attempts):
        size = n or rng.randint(min_entities, max_entities)
        if size < 3:
            size = 3
        people = rng.sample(PERSON_NAMES, size)
        domains = rng.sample(DOMAIN_VALUES, size)
        oses = rng.sample(OS_VALUES, size)

        dperm = rng.sample(range(1, size + 1), size)
        domperm = rng.sample(domains, size)
        osperm = rng.sample(oses, size)
        solution = {
            d: {"person": p, "domain": domperm[i], "os": osperm[i]}
            for i, (p, d) in enumerate(zip(people, dperm))
        }

        candidates = _all_true_clues(solution, size, rng)
        selected = _select_minimal_unique(candidates, people, domains, oses, size, solution, rng, budget)
        if selected is None:
            continue

        if not insufficient:
            return Puzzle(size, people, domains, oses, solution, selected, "unique", seed=seed)

        # insufficient: drop the last clue so the solution is no longer unique,
        # then pick a query whose value genuinely varies.
        if len(selected) < 3:
            continue
        reduced = selected[:-1]
        try:
            sols = solve(people, domains, oses, reduced, size, limit=8, node_budget=budget)
        except BudgetExceeded:
            continue
        if len(sols) < 2:
            continue
        query = None
        for d in range(1, size + 1):
            for cat in CATEGORIES:
                vals = {s[d][cat] for s in sols}
                if len(vals) > 1:
                    query = f"desk {d} {cat}"
                    break
            if query:
                break
        if query is None:
            continue
        return Puzzle(size, people, domains, oses, solution, reduced, "insufficient", query=query, seed=seed)
    return None


# ==============================================================================
# Validation (Stage 3)
# ==============================================================================
_INSUFFICIENT_MARKERS = (
    "not enough information",
    "insufficient",
    "cannot be determined",
    "can't be determined",
    "undetermined",
    "not determinable",
)


def validate_answer(puzzle: Puzzle, text: str) -> bool:
    """Return True iff ``text`` contains the correct final answer.

    For ``unique`` puzzles every person must appear on a line that also carries
    their correct desk number, domain and OS (matches the requested markdown
    table / consolidated bullet layouts). For ``insufficient`` puzzles the text
    must state that the answer cannot be determined.
    """
    low = text.lower()
    if puzzle.answer_kind == "insufficient":
        return any(m in low for m in _INSUFFICIENT_MARKERS)

    lines = text.splitlines()
    for d in range(1, puzzle.n + 1):
        a = puzzle.solution[d]
        person, dom, os_ = a["person"], a["domain"], a["os"]
        ok = False
        for ln in lines:
            if person.lower() in ln.lower() and str(d) in ln and dom.lower() in ln.lower() and os_.lower() in ln.lower():
                ok = True
                break
        if not ok:
            return False
    return True


def self_test() -> None:
    """Cheap invariant checks (used by tests and ``--self_test``)."""
    for seed in range(20):
        p = generate_puzzle(seed, n=4)
        assert p is not None, f"failed to generate puzzle seed={seed}"
        sols = solve(p.people, p.domain_values, p.os_values, p.clues, p.n, limit=2)
        assert len(sols) == 1 and _same_solution(sols[0], p.solution), f"non-unique seed={seed}"
        assert validate_answer(p, p.render_table()), f"self-validate failed seed={seed}"


if __name__ == "__main__":  # pragma: no cover
    self_test()
    demo = generate_puzzle(1234, n=5)
    assert demo is not None
    print(demo.render_problem())
    print()
    print(demo.render_trace())
