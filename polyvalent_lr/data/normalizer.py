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
    def parse_proof_intermediates(cls, proof_with_intermediates: List[Dict[str, Any]]) -> str:
        """Parses ProofWriter's proofsWithIntermediates into human-readable multi-step reasoning traces."""
        if not proof_with_intermediates:
            return ""
        
        trace_lines = []
        for p_obj in proof_with_intermediates:
            intermediates = p_obj.get("intermediates", {})
            rep = p_obj.get("representation", "")
            if intermediates and isinstance(intermediates, dict):
                for int_key, int_val in intermediates.items():
                    int_text = int_val.get("text", "") if isinstance(int_val, dict) else str(int_val)
                    trace_lines.append(f"Derived intermediate fact: {int_text}")
            trace_lines.append(f"Proof derivation structure: {rep}")
        return "\n".join(trace_lines)

    @classmethod
    def normalize_proofwriter(cls, item: Dict[str, Any]) -> List[LogicalExample]:
        """Normalizes an official ProofWriter meta-*.jsonl record containing a theory and multiple questions.
        
        Returns a list of LogicalExample instances (one per question in the theory).
        """
        theory_id = item.get("id", "pw_theory_unknown")
        
        # 1. Extract premises (triples + rules, sentences in NatLang, or raw theory text)
        premises = []
        if "sentences" in item and isinstance(item["sentences"], dict):
            premises = [s.strip() for s in item["sentences"].values() if s.strip()]
        elif "triples" in item or "rules" in item:
            if "triples" in item and isinstance(item["triples"], dict):
                for t in item["triples"].values():
                    t_text = t.get("text", "").strip() if isinstance(t, dict) else str(t).strip()
                    if t_text:
                        premises.append(t_text)
            if "rules" in item and isinstance(item["rules"], dict):
                for r in item["rules"].values():
                    r_text = r.get("text", "").strip() if isinstance(r, dict) else str(r).strip()
                    if r_text:
                        premises.append(r_text)
        elif "theory" in item:
            th = item["theory"]
            if isinstance(th, list):
                premises = [p.strip() for p in th if p.strip()]
            else:
                premises = [p.strip() + "." for p in th.split(".") if p.strip()]

        # 2. Extract questions (official ProofWriter has a questions dictionary {Q1: ..., Q2: ...})
        examples = []
        questions_dict = item.get("questions", {})

        # If flat single-question item (legacy / starter format)
        if not questions_dict and ("question" in item or "hypothesis" in item):
            q_text = item.get("question", item.get("hypothesis", ""))
            ans_raw = item.get("answer", "Unknown")
            ex = LogicalExample(
                id=f"pw_{theory_id}_single",
                group_id=f"pw_theory_{theory_id}",
                source=DatasetSource.PROOFWRITER,
                task_type=LogicTaskType.DEDUCTION,
                premises=premises,
                hypothesis_or_query=q_text,
                proof_trace_text=item.get("proof", ""),
                label=cls.normalize_label(ans_raw),
                metadata={"depth": item.get("depth", 0), "strategy": item.get("strategy", "proof")}
            )
            return [ex]

        # Process all questions in the theory
        for q_id, q_data in questions_dict.items():
            if not isinstance(q_data, dict):
                continue
            
            q_text = q_data.get("question", "")
            ans_raw = q_data.get("answer", "Unknown")
            q_dep = q_data.get("QDep", 0)
            strategy = q_data.get("strategy", "proof")
            
            # Format proof trace
            proofs_with_int = q_data.get("proofsWithIntermediates", [])
            proof_trace = cls.parse_proof_intermediates(proofs_with_int)
            if not proof_trace:
                proof_trace = str(q_data.get("proofs", ""))

            example = LogicalExample(
                id=f"pw_{theory_id}_{q_id}",
                group_id=f"pw_theory_{theory_id}",
                source=DatasetSource.PROOFWRITER,
                task_type=LogicTaskType.DEDUCTION,
                premises=premises,
                hypothesis_or_query=q_text,
                proof_trace_text=proof_trace if proof_trace else None,
                label=cls.normalize_label(ans_raw),
                metadata={
                    "theory_id": theory_id,
                    "question_id": q_id,
                    "depth": q_dep,
                    "strategy": strategy,
                    "maxD": item.get("maxD", 0),
                    "NFact": item.get("NFact", 0),
                    "NRule": item.get("NRule", 0)
                }
            )
            examples.append(example)

        return examples

    @classmethod
    def normalize_proofwriter_abduct(cls, item: Dict[str, Any]) -> List[LogicalExample]:
        """Normalizes official ProofWriter meta-abduct-*.jsonl records into abductive reasoning examples."""
        theory_id = item.get("id", "abduct_theory_unknown")
        
        premises = []
        if "triples" in item and isinstance(item["triples"], dict):
            for t in item["triples"].values():
                t_text = t.get("text", "").strip() if isinstance(t, dict) else str(t).strip()
                if t_text:
                    premises.append(t_text)
        if "rules" in item and isinstance(item["rules"], dict):
            for r in item["rules"].values():
                r_text = r.get("text", "").strip() if isinstance(r, dict) else str(r).strip()
                if r_text:
                    premises.append(r_text)

        examples = []
        abductions_dict = item.get("abductions", {})
        
        for mf_id, mf_data in abductions_dict.items():
            if not isinstance(mf_data, dict):
                continue
            
            obs_question = mf_data.get("question", "")
            answers = mf_data.get("answers", [])
            
            # For abductive reasoning: Given rules + facts + observation -> predict missing fact
            for idx, ans_obj in enumerate(answers):
                missing_fact = ans_obj.get("text", "")
                proof_str = ans_obj.get("proof", "")
                q_dep = ans_obj.get("QDep", 1)

                ex = LogicalExample(
                    id=f"pw_abduct_{theory_id}_{mf_id}_{idx}",
                    group_id=f"pw_abduct_theory_{theory_id}",
                    source=DatasetSource.ABDUCTION_RULES,
                    task_type=LogicTaskType.ABDUCTION,
                    premises=premises + [f"Observed: {obs_question}"],
                    hypothesis_or_query=f"Hypothesize missing fact: {missing_fact}",
                    proof_trace_text=f"Missing fact '{missing_fact}' completes proof: {proof_str}",
                    label="True",
                    metadata={"depth": q_dep, "target_question": obs_question, "missing_fact": missing_fact}
                )
                examples.append(ex)

        return examples

    @classmethod
    def normalize_folio(cls, item: Dict[str, Any]) -> LogicalExample:
        """Normalizes an official Yale-LILY FOLIO record from data/v0.0/."""
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
        
        formal_annotations = item.get("premises-FOL", item.get("premises_FOL", []))
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
            metadata={"story_id": group_id, "example_id": item.get("example_id", 0)}
        )

    @classmethod
    def normalize_p_folio(cls, item: Dict[str, Any]) -> LogicalExample:
        """Normalizes a raw P-FOLIO record containing explicit human proof annotations."""
        raw_id = item.get("story_id", item.get("id", "pfolio_unknown"))
        group_id = str(raw_id)
        
        premises = item.get("premises", [])
        if isinstance(premises, str):
            premises = [p.strip() for p in premises.split("\n") if p.strip()]
        else:
            premises = [str(p).strip() for p in premises if str(p).strip()]
            
        hypothesis = item.get("conclusion", item.get("hypothesis", ""))
        raw_label = item.get("label", "")
        normalized_label = cls.normalize_label(raw_label)
        
        proof_trace = item.get("proof_trace", item.get("annotated_proof", item.get("explanation", "")))
        
        return LogicalExample(
            id=f"pfolio_{raw_id}_{item.get('example_id', 0)}",
            group_id=f"pfolio_story_{group_id}",
            source=DatasetSource.P_FOLIO,
            task_type=LogicTaskType.DEDUCTION,
            premises=premises,
            hypothesis_or_query=hypothesis,
            proof_trace_text=proof_trace if proof_trace else None,
            label=normalized_label,
            metadata={"story_id": group_id, "example_id": item.get("example_id", 0)}
        )

    @classmethod
    def normalize_abduction_rules(cls, item: Dict[str, Any]) -> List[LogicalExample]:
        """Normalizes official LogiTorch / AbductionRules records.
        
        Given a context of rules/facts and an observation ('text'), the goal is to infer the 
        missing explanation ('label').
        """
        raw_id = item.get("id", "abduct_0")
        context_str = item.get("context", "")
        if not context_str and "premises" in item:
            context_str = " ".join(item["premises"]) if isinstance(item["premises"], list) else str(item["premises"])
            
        # Parse context into premise sentences
        if context_str:
            premises = [p.strip() + "." for p in context_str.split(".") if p.strip()]
        else:
            premises = [str(p).strip() for p in item.get("context_rules", []) + item.get("observations", []) if str(p).strip()]

        group_id = f"abduct_ctx_{raw_id}"

        examples = []
        questions = item.get("questions", [])

        if questions and isinstance(questions, list):
            for idx, q_obj in enumerate(questions):
                q_id = q_obj.get("id", f"{raw_id}-Q{idx+1}")
                obs_text = q_obj.get("text", "")
                abducted_explanation = q_obj.get("label", "")
                q_cat = q_obj.get("QCat", "0")

                ex = LogicalExample(
                    id=f"abduct_{q_id}",
                    group_id=group_id,
                    source=DatasetSource.ABDUCTION_RULES,
                    task_type=LogicTaskType.ABDUCTION,
                    premises=premises + [f"Observed fact: {obs_text}"],
                    hypothesis_or_query=f"Hypothesize the missing explanation: {abducted_explanation}",
                    proof_trace_text=f"Given observation '{obs_text}' and context rules, the missing explanation is: '{abducted_explanation}'.",
                    label=abducted_explanation if abducted_explanation in ["True", "False", "Uncertain"] else "True",
                    metadata={
                        "theory_id": raw_id,
                        "question_id": q_id,
                        "observed_text": obs_text,
                        "abducted_explanation": abducted_explanation,
                        "QCat": q_cat
                    }
                )
                examples.append(ex)
        else:
            # Flat format fallback
            q_text = item.get("question", item.get("candidate_hypothesis", item.get("explanation", "")))
            q_label = item.get("label", item.get("plausibility", "True"))
            ex = LogicalExample(
                id=f"abduct_{raw_id}_0",
                group_id=group_id,
                source=DatasetSource.ABDUCTION_RULES,
                task_type=LogicTaskType.ABDUCTION,
                premises=premises,
                hypothesis_or_query=q_text,
                proof_trace_text=item.get("abductive_chain", None),
                label=cls.normalize_label(q_label),
                metadata={"context_id": raw_id}
            )
            examples.append(ex)

        return examples

    @classmethod
    def normalize_logicbench(cls, item: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> LogicalExample:
        """Normalizes official LogicBench(Eval) BQA/MCQA records."""
        meta = metadata or {}
        raw_id = item.get("id", item.get("question_id", "lb_unknown"))
        pattern_id = item.get("pattern_id", item.get("pattern", "pattern_0"))
        
        context_str = item.get("context", item.get("premise", ""))
        if isinstance(context_str, list):
            premises = [p.strip() for p in context_str if p.strip()]
        else:
            premises = [p.strip() + "." for p in str(context_str).split(".") if p.strip()]

        hypothesis = item.get("question", item.get("hypothesis", ""))
        raw_label = item.get("answer", item.get("label", "True"))
        
        # In MCQA, append options to hypothesis if present
        if "options" in item and isinstance(item["options"], (list, dict)):
            options_str = "\nOptions:\n" + "\n".join(f"- {k if isinstance(item['options'], dict) else i+1}: {v}" for i, (k, v) in enumerate(item["options"].items() if isinstance(item["options"], dict) else enumerate(item["options"])))
            hypothesis = f"{hypothesis}\n{options_str}"

        return LogicalExample(
            id=f"lb_{pattern_id}_{raw_id}",
            group_id=f"lb_pattern_{pattern_id}",
            source=DatasetSource.PROOFWRITER,  # placeholder benchmark source
            task_type=LogicTaskType.DEDUCTION,
            premises=premises,
            hypothesis_or_query=hypothesis,
            proof_trace_text=item.get("reasoning", item.get("proof", None)),
            label=cls.normalize_label(raw_label),
            metadata={"pattern_id": pattern_id, **meta}
        )

    @classmethod
    def normalize_multi_logieval(cls, item: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> LogicalExample:
        """Normalizes official Multi-LogiEval multi-depth rule records (d1_Data to d5_Data)."""
        meta = metadata or {}
        raw_id = item.get("id", "mle_unknown")
        depth = meta.get("depth", item.get("depth", 1))
        
        context_str = item.get("context", item.get("premises", ""))
        if isinstance(context_str, list):
            premises = [p.strip() for p in context_str if p.strip()]
        else:
            premises = [p.strip() + "." for p in str(context_str).split(".") if p.strip()]

        hypothesis = item.get("question", item.get("hypothesis", item.get("conclusion", "")))
        raw_label = item.get("answer", item.get("label", "True"))

        return LogicalExample(
            id=f"mle_d{depth}_{raw_id}",
            group_id=f"mle_depth_{depth}_{raw_id}",
            source=DatasetSource.FOLIO,  # placeholder benchmark source
            task_type=LogicTaskType.DEDUCTION,
            premises=premises,
            hypothesis_or_query=hypothesis,
            proof_trace_text=item.get("reasoning", item.get("proof", None)),
            label=cls.normalize_label(raw_label),
            metadata={"depth": depth, **meta}
        )

    @classmethod
    def normalize_logicnli(cls, item: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> LogicalExample:
        """Normalizes official LogicNLI language/logic diagnostic test records."""
        meta = metadata or {}
        raw_id = item.get("id", "lnli_unknown")
        group_id = item.get("group", raw_id)
        
        premise_str = item.get("premise", "")
        if isinstance(premise_str, list):
            premises = [p.strip() for p in premise_str if p.strip()]
        else:
            premises = [p.strip() + "." for p in str(premise_str).split(".") if p.strip()]

        hypothesis = item.get("hypothesis", "")
        raw_label = item.get("label", "neutral")
        
        # NLI standard mapping: entailment -> True, contradiction -> False, neutral -> Uncertain
        normalized_label = cls.normalize_label(raw_label)

        return LogicalExample(
            id=f"lnli_{raw_id}",
            group_id=f"lnli_grp_{group_id}",
            source=DatasetSource.FOLIO,
            task_type=LogicTaskType.DEDUCTION,
            premises=premises,
            hypothesis_or_query=hypothesis,
            proof_trace_text=None,
            label=normalized_label,
            metadata={"variant": meta.get("variant", "language"), **meta}
        )

