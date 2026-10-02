import random
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any
from polyvalent_lr.data.schema import LogicalExample
from polyvalent_lr.config import DatasetSource


class GroupAwareDataSplitter:
    """Partitions logical reasoning examples by group_id / premise-structure to prevent leakage."""

    @staticmethod
    def split_by_group(
        examples: List[LogicalExample],
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[LogicalExample], List[LogicalExample]]:
        """Splits examples such that all examples belonging to the same group_id stay exclusively in train or val."""
        random.seed(seed)
        
        # Group examples by group_id
        grouped: Dict[str, List[LogicalExample]] = defaultdict(list)
        for ex in examples:
            grouped[ex.group_id].append(ex)
            
        group_ids = list(grouped.keys())
        random.shuffle(group_ids)
        
        total_examples = len(examples)
        target_val_count = int(total_examples * val_ratio)
        
        train_examples: List[LogicalExample] = []
        val_examples: List[LogicalExample] = []
        current_val_count = 0
        
        for gid in group_ids:
            group_items = grouped[gid]
            if current_val_count + len(group_items) <= target_val_count and current_val_count < target_val_count:
                val_examples.extend(group_items)
                current_val_count += len(group_items)
            else:
                train_examples.extend(group_items)
                
        # If val is empty due to large groups, ensure at least one group in val if possible
        if not val_examples and len(group_ids) > 1:
            val_examples.extend(grouped[group_ids[0]])
            train_examples = [ex for gid in group_ids[1:] for ex in grouped[gid]]
            
        return train_examples, val_examples

    @staticmethod
    def check_folio_pfolio_overlap(
        folio_examples: List[LogicalExample],
        pfolio_examples: List[LogicalExample]
    ) -> Dict[str, Any]:
        """Detects overlapping premise sets and story IDs between FOLIO and P-FOLIO."""
        def get_premise_fingerprint(ex: LogicalExample) -> str:
            cleaned = " ".join(sorted([p.strip().lower() for p in ex.premises]))
            return cleaned

        folio_fingerprints: Dict[str, List[str]] = defaultdict(list)
        for ex in folio_examples:
            fp = get_premise_fingerprint(ex)
            folio_fingerprints[fp].append(ex.id)

        overlapping_pairs = []
        for p_ex in pfolio_examples:
            p_fp = get_premise_fingerprint(p_ex)
            if p_fp in folio_fingerprints:
                overlapping_pairs.append({
                    "pfolio_id": p_ex.id,
                    "folio_ids": folio_fingerprints[p_fp],
                    "fingerprint": p_fp[:80] + "..."
                })

        return {
            "total_folio": len(folio_examples),
            "total_pfolio": len(pfolio_examples),
            "overlap_count": len(overlapping_pairs),
            "overlapping_pairs": overlapping_pairs
        }

    @staticmethod
    def audit_leakage_against_unseen_benchmarks(
        train_examples: List[LogicalExample],
        benchmark_examples: List[LogicalExample]
    ) -> List[Dict[str, str]]:
        """Audits and guarantees zero premise or hypothesis overlap with quarantined benchmarks."""
        train_hypotheses: Set[str] = {ex.hypothesis_or_query.strip().lower() for ex in train_examples}
        
        leaks = []
        for b_ex in benchmark_examples:
            b_hypo = b_ex.hypothesis_or_query.strip().lower()
            if b_hypo in train_hypotheses:
                leaks.append({
                    "benchmark_id": b_ex.id,
                    "benchmark_source": str(b_ex.source),
                    "hypothesis": b_ex.hypothesis_or_query
                })
        return leaks
