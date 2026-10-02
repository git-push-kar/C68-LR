import re
from typing import Dict, Any, List
from polyvalent_lr.config import DatasetSource, LogicTaskType
from polyvalent_lr.data.schema import LogicalExample, ProofStep


class DatasetNormalizer:
    """Normalizes heterogeneous reasoning datasets into a unified LogicalExample representation."""

    @staticmethod
    def normalize_label(raw_label: Any) -> str:
        """Standardizes task labels across datasets into True/False/Uncertain format."""
        if isinstance(raw_label, bool):
            return "True" if raw_label else "False"
        
        s = str(raw_label).strip().lower()
        if s in ["true", "entailed", "entailment", "1", "yes", "valid", "proved"]:
            return "True"
        elif s in ["false", "contradiction", "contradicted", "0", "no", "invalid", "disproved"]:
            return "False"
        elif s in ["uncertain", "neutral", "unknown", "undetermined", "unknown_or_neutral"]:
            return "Uncertain"
        return s.capitalize()

    @classmethod
    def normalize_proofwriter(cls, item: Dict[str, Any]) -> LogicalExample:
        """Normalizes a raw ProofWriter record."""
        # Typical ProofWriter format: theory (facts/rules), question/hypothesis, answer, proof
        raw_id = item.get("id", "pw_unknown")
        # In ProofWriter, theory usually corresponds to a scenario group
        group_id = item.get("theory_id", raw_id.split("-")[0] if "-" in str(raw_id) else raw_id)
        
        theory = item.get("theory", "")
        if isinstance(theory, list):
            premises = [p.strip() for p in theory if p.strip()]
        else:
            premises = [p.strip() for p in theory.split(".") if p.strip()]
            
        hypothesis = item.get("question", item.get("hypothesis", ""))
        raw_answer = item.get("answer", "")
        normalized_label = cls.normalize_label(raw_answer)
        
        # Parse proof steps if available
        proof_text = item.get("proof", "")
        proof_steps = []
        if proof_text:
            lines = [l.strip() for l in proof_text.split(";") if l.strip()]
            for idx, line in enumerate(lines, 1):
                proof_steps.append(ProofStep(
                    step_id=idx,
                    rule_or_fact_used=line,
                    intermediate_conclusion=f"Inference step {idx}"
                ))

        return LogicalExample(
            id=f"pw_{raw_id}",
            group_id=f"pw_group_{group_id}",
            source=DatasetSource.PROOFWRITER,
            task_type=LogicTaskType.DEDUCTION,
            premises=premises,
            hypothesis_or_query=hypothesis,
            proof_steps=proof_steps if proof_steps else None,
            proof_trace_text=proof_text if proof_text else None,
            label=normalized_label,
            metadata={"depth": item.get("depth", 0), "strategy": item.get("strategy", "deduction")}
        )

    @classmethod
    def normalize_folio(cls, item: Dict[str, Any]) -> LogicalExample:
        """Normalizes a raw FOLIO record."""
        raw_id = item.get("story_id", item.get("id", "folio_unknown"))
        group_id = str(raw_id)
        
        premises_text = item.get("premises", [])
        if isinstance(premises_text, str):
            premises = [p.strip() for p in premises_text.split("\n") if p.strip()]
        else:
            premises = [str(p).strip() for p in premises_text if str(p).strip()]
            
        hypothesis = item.get("conclusion", item.get("hypothesis", ""))
        raw_label = item.get("label", "")
        normalized_label = cls.normalize_label(raw_label)
        
        formal_annotations = item.get("premises_FOL", [])
        if isinstance(formal_annotations, str):
            formal_annotations = [formal_annotations]

        return LogicalExample(
            id=f"folio_{raw_id}_{item.get('example_id', 0)}",
            group_id=f"folio_story_{group_id}",
            source=DatasetSource.FOLIO,
            task_type=LogicTaskType.DEDUCTION,
            premises=premises,
            hypothesis_or_query=hypothesis,
            formal_annotations=formal_annotations,
            proof_trace_text=item.get("explanation", None),
            label=normalized_label,
            metadata={"story_id": group_id}
        )

    @classmethod
    def normalize_p_folio(cls, item: Dict[str, Any]) -> LogicalExample:
        """Normalizes a raw P-FOLIO record containing explicit proof annotations."""
        raw_id = item.get("id", item.get("story_id", "pfolio_unknown"))
        group_id = str(item.get("story_id", raw_id))
        
        premises = item.get("premises", [])
        if isinstance(premises, str):
            premises = [p.strip() for p in premises.split("\n") if p.strip()]
            
        hypothesis = item.get("conclusion", item.get("hypothesis", ""))
        raw_label = item.get("label", "")
        normalized_label = cls.normalize_label(raw_label)
        
        # P-FOLIO contains annotated human proof traces
        proof_trace = item.get("proof_trace", item.get("annotated_proof", ""))
        
        return LogicalExample(
            id=f"pfolio_{raw_id}",
            group_id=f"pfolio_story_{group_id}",
            source=DatasetSource.P_FOLIO,
            task_type=LogicTaskType.DEDUCTION,
            premises=premises,
            hypothesis_or_query=hypothesis,
            proof_trace_text=proof_trace if proof_trace else None,
            label=normalized_label,
            metadata={"annotator_confidence": item.get("confidence", 1.0)}
        )

    @classmethod
    def normalize_abduction_rules(cls, item: Dict[str, Any]) -> LogicalExample:
        """Normalizes a raw AbductionRules record."""
        raw_id = item.get("id", "abduct_unknown")
        group_id = item.get("rule_context_id", raw_id)
        
        rules_and_obs = item.get("context_rules", []) + item.get("observations", [])
        if isinstance(rules_and_obs, str):
            premises = [p.strip() for p in rules_and_obs.split("\n") if p.strip()]
        else:
            premises = [str(p).strip() for p in rules_and_obs if str(p).strip()]
            
        hypothesis = item.get("candidate_hypothesis", item.get("explanation", ""))
        raw_label = item.get("plausibility", item.get("label", "True"))
        normalized_label = cls.normalize_label(raw_label)
        
        return LogicalExample(
            id=f"abduct_{raw_id}",
            group_id=f"abduct_ctx_{group_id}",
            source=DatasetSource.ABDUCTION_RULES,
            task_type=LogicTaskType.ABDUCTION,
            premises=premises,
            hypothesis_or_query=hypothesis,
            proof_trace_text=item.get("abductive_chain", None),
            label=normalized_label,
            metadata={"rule_type": "abductive"}
        )
