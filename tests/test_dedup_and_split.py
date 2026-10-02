import pytest
from polyvalent_lr.data.schema import LogicalExample, LogicTaskType
from polyvalent_lr.config import DatasetSource, UnseenBenchmark
from polyvalent_lr.data.dedup import GroupAwareDataSplitter


def test_group_aware_split():
    examples = [
        LogicalExample(
            id=f"ex_{i}_{j}",
            group_id=f"group_{i}",
            source=DatasetSource.PROOFWRITER,
            task_type=LogicTaskType.DEDUCTION,
            premises=[f"Fact {i}"],
            hypothesis_or_query=f"Query {j}",
            label="True"
        )
        for i in range(10)
        for j in range(3)
    ]
    train, val = GroupAwareDataSplitter.split_by_group(examples, val_ratio=0.3, seed=42)

    train_groups = {e.group_id for e in train}
    val_groups = {e.group_id for e in val}

    # Strict isolation check: intersection of group_ids must be empty
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train) + len(val) == len(examples)


def test_leakage_audit():
    train_ex = [
        LogicalExample(
            id="t1",
            group_id="g1",
            source=DatasetSource.PROOFWRITER,
            task_type=LogicTaskType.DEDUCTION,
            premises=["A is true"],
            hypothesis_or_query="Is A true?",
            label="True"
        )
    ]
    bench_leaked = [
        LogicalExample(
            id="b1",
            group_id="bg1",
            source=DatasetSource.PROOFWRITER,
            task_type=LogicTaskType.DEDUCTION,
            premises=["A is true"],
            hypothesis_or_query="Is A true?",
            label="True"
        )
    ]
    bench_clean = [
        LogicalExample(
            id="b2",
            group_id="bg2",
            source=DatasetSource.PROOFWRITER,
            task_type=LogicTaskType.DEDUCTION,
            premises=["B is true"],
            hypothesis_or_query="Is B true?",
            label="True"
        )
    ]

    leaks = GroupAwareDataSplitter.audit_leakage_against_unseen_benchmarks(train_ex, bench_leaked)
    assert len(leaks) == 1
    assert leaks[0]["benchmark_id"] == "b1"

    no_leaks = GroupAwareDataSplitter.audit_leakage_against_unseen_benchmarks(train_ex, bench_clean)
    assert len(no_leaks) == 0
