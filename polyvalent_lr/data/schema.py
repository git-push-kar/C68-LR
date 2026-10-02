from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from polyvalent_lr.config import LogicTaskType, DatasetSource


class ProofStep(BaseModel):
    step_id: int
    rule_or_fact_used: str
    intermediate_conclusion: str


class LogicalExample(BaseModel):
    id: str = Field(..., description="Unique example ID")
    group_id: str = Field(..., description="Premise/Problem group ID for leakage-free splitting")
    source: DatasetSource = Field(..., description="Origin dataset")
    task_type: LogicTaskType = Field(default=LogicTaskType.DEDUCTION, description="Deduction, Abduction, etc.")
    premises: List[str] = Field(..., description="List of premises, rules, and facts")
    hypothesis_or_query: str = Field(..., description="The query statement, question, or hypothesis to evaluate")
    formal_annotations: Optional[List[str]] = Field(default=None, description="Optional FOL / symbolic representations")
    proof_steps: Optional[List[ProofStep]] = Field(default=None, description="Step-by-step reasoning or formal proof")
    proof_trace_text: Optional[str] = Field(default=None, description="Human/natural language reasoning trace")
    label: str = Field(..., description="Normalized target answer (e.g. True, False, Uncertain / Entailed, Contradicted, Neutral)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_instruction_prompt(self, style: str = "structured_cot") -> str:
        """Formats the logical example into an instruction prompt."""
        premises_text = "\n".join(f"- {p}" for p in self.premises)
        
        if self.task_type == LogicTaskType.DEDUCTION:
            task_instruction = (
                "Given the following facts and rules (premises), determine whether the hypothesis is True, False, or Uncertain. "
                "Provide a step-by-step logical proof trace before giving the final answer."
            )
        elif self.task_type == LogicTaskType.ABDUCTION:
            task_instruction = (
                "Given the observations and background rules, infer the most plausible explanation (hypothesis). "
                "Provide a step-by-step abductive reasoning trace before stating your conclusion."
            )
        else:
            task_instruction = "Analyze the premises and determine the validity of the hypothesis with a proof trace."

        prompt = (
            f"<|im_start|>system\nYou are a rigorous logical reasoning engine. "
            f"Solve the problem by validating rules and facts systematically.<|im_end|>\n"
            f"<|im_start|>user\n{task_instruction}\n\n"
            f"Premises:\n{premises_text}\n\n"
            f"Hypothesis:\n{self.hypothesis_or_query}\n\n"
            f"Format your response as:\n"
            f"<think>\n[Step-by-step deduction/proof steps]\n</think>\n"
            f"<answer>[Final Answer]</answer><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def to_target_text(self) -> str:
        """Formats the target ground truth response with reasoning trace and final label."""
        if self.proof_trace_text:
            trace = self.proof_trace_text.strip()
        elif self.proof_steps:
            trace = "\n".join(
                f"Step {s.step_id}: Apply '{s.rule_or_fact_used}' -> Derived: {s.intermediate_conclusion}"
                for s in self.proof_steps
            )
        else:
            trace = "Deduction derived directly from given premises."

        return f"<think>\n{trace}\n</think>\n<answer>{self.label.strip()}</answer><|im_end|>"
