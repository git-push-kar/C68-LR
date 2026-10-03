import torch
from typing import Dict, Any, Tuple, Optional
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from polyvalent_lr.config import LoRAConfig
from polyvalent_lr.models.loader import load_internvl_model_and_tokenizer


def create_peft_config(config: LoRAConfig) -> LoraConfig:
    """Constructs a PEFT LoraConfig for InternVL3-2B language model layers."""
    return LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
    )


def setup_internvl_lr_lora(
    model_name_or_path: str,
    lora_config: LoRAConfig,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    gradient_checkpointing: bool = True
) -> Tuple[Any, Any]:
    """Loads InternVL3-2B language backbone and attaches LR LoRA adapters with GPU acceleration."""
    # Load base model using unified, version-compatible loader
    model, tokenizer = load_internvl_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        is_eval=False
    )

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[LoRA Setup] Enabled gradient checkpointing for VRAM efficiency.")

    # Freeze all base parameters (including vision encoder)
    for param in model.parameters():
        param.requires_grad = False

    peft_conf = create_peft_config(lora_config)
    peft_model = get_peft_model(model, peft_conf)

    trainable_params, all_param = peft_model.get_nb_trainable_parameters()
    print(
        f"[LoRA Setup] Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable ratio: {100 * trainable_params / all_param:.4f}%"
    )

    return peft_model, tokenizer
