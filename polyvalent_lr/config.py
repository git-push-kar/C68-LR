from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class LogicTaskType(str, Enum):
    DEDUCTION = "deduction"
    ABDUCTION = "abduction"
    INDUCTION = "induction"


class DatasetSource(str, Enum):
    PROOFWRITER = "ProofWriter"
    P_FOLIO = "P-FOLIO"
    FOLIO = "FOLIO"
    ABDUCTION_RULES = "AbductionRules"


class UnseenBenchmark(str, Enum):
    LOGICBENCH = "LogicBench"
    MULTI_LOGIEVAL = "Multi-LogiEval"
    LOGICNLI = "LogicNLI"


class LoRAConfig(BaseModel):
    r: int = Field(default=32, description="LoRA rank")
    lora_alpha: int = Field(default=64, description="LoRA scaling factor alpha")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout rate")
    target_modules: List[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target attention and MLP weight matrices in the language model",
    )
    bias: str = Field(default="none", description="LoRA bias setting")
    task_type: str = Field(default="CAUSAL_LM", description="PEFT task type")


class CurriculumStageConfig(BaseModel):
    stage_id: int
    name: str
    datasets: List[DatasetSource]
    epochs: int = 2
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    description: str


class DataPipelineConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    val_split_ratio: float = 0.1
    random_seed: int = 42
    check_folio_pfolio_overlap: bool = True
    normalize_formats: bool = True


class EvaluationConfig(BaseModel):
    benchmarks: List[UnseenBenchmark] = [
        UnseenBenchmark.LOGICBENCH,
        UnseenBenchmark.MULTI_LOGIEVAL,
        UnseenBenchmark.LOGICNLI,
    ]
    max_eval_samples_per_bench: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    prompt_template_style: str = "structured_cot"


class LRModuleConfig(BaseModel):
    base_model_name_or_path: str = "OpenGVLab/InternVL3-2B"
    output_dir: str = "checkpoints/internvl3_2b_lr"
    merged_output_dir: str = "checkpoints/internvl3_2b_lr_fused"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    data: DataPipelineConfig = Field(default_factory=DataPipelineConfig)
    curriculum_stages: List[CurriculumStageConfig] = Field(
        default_factory=lambda: [
            CurriculumStageConfig(
                stage_id=1,
                name="structured_deduction",
                datasets=[DatasetSource.PROOFWRITER],
                epochs=2,
                learning_rate=2e-4,
                description="Stage 1: Multi-step structured deduction from rules and facts.",
            ),
            CurriculumStageConfig(
                stage_id=2,
                name="proof_and_nl_fol",
                datasets=[DatasetSource.P_FOLIO, DatasetSource.FOLIO],
                epochs=2,
                learning_rate=1e-4,
                description="Stage 2: Natural language First-Order Logic & human proof supervision.",
            ),
            CurriculumStageConfig(
                stage_id=3,
                name="abductive_reasoning_optional",
                datasets=[DatasetSource.ABDUCTION_RULES],
                epochs=1,
                learning_rate=5e-5,
                description="Stage 3 (Optional): Abductive inference from observations to hypotheses.",
            ),
        ]
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
