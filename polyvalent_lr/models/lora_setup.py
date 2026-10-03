import torch
from typing import Dict, Any, Tuple, Optional
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
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    gradient_checkpointing: bool = True
) -> Tuple[Any, Any]:
    """Loads InternVL3-2B language backbone and attaches LR LoRA adapters with GPU acceleration."""
    if device_map is None:
        device_map = "auto" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

    # Hardware & placement diagnosis
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_avail else 0.0

    print(f"\n{'='*60}")
    print(f"[LoRA Setup] Target Device: {device_name} | CUDA: {cuda_avail} | VRAM: {vram_gb:.2f} GB")
    print(f"[LoRA Setup] Model: {model_name_or_path} | Precision: {torch_dtype} | device_map: '{device_map}'")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
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
