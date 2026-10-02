import torch
from typing import Dict, Any, Tuple
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from polyvalent_lr.config import LoRAConfig


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
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16
) -> Tuple[Any, Any]:
    """Loads InternVL3-2B language backbone and attaches LR LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Ensure vision encoder and non-targeted layers are frozen
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
