import torch
import inspect
import functools
import transformers
from typing import Tuple, Any, Optional
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


# ==============================================================================
# Compatibility Patch for Transformers >= 4.48 with InternVL Remote Code
# ==============================================================================
def _apply_tied_weights_patch():
    try:
        from torch import nn
        
        # Patch PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = property(
                lambda self: getattr(self, "_tied_weights_keys", {})
                if isinstance(getattr(self, "_tied_weights_keys", {}), dict)
                else {k: None for k in getattr(self, "_tied_weights_keys", [])}
                if isinstance(getattr(self, "_tied_weights_keys", None), (list, tuple, set))
                else {}
            )

        # Patch nn.Module as a universal fallback for custom model classes
        if not hasattr(nn.Module, "all_tied_weights_keys"):
            @property
            def all_tied_weights_keys(self):
                tied = getattr(self, "_tied_weights_keys", {})
                if isinstance(tied, dict):
                    return tied
                elif isinstance(tied, (list, tuple, set)):
                    return {k: None for k in tied}
                return {}
            nn.Module.all_tied_weights_keys = all_tied_weights_keys
    except Exception as e:
        print(f"[Warning] Could not apply tied weights compatibility patch: {e}")

_apply_tied_weights_patch()


def _patch_model_forward(model: Any) -> None:
    """
    Wraps model.forward to safely filter keyword arguments (e.g., 'inputs_embeds',
    'num_items_in_batch') that are passed by PEFT/Trainer but not accepted by InternVLChatModel.
    """
    cls = model.__class__
    if getattr(cls, "_forward_kwargs_patched", False):
        return

    orig_forward = cls.forward
    sig = inspect.signature(orig_forward)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if not has_var_keyword:
        valid_kwargs = set(sig.parameters.keys())

        @functools.wraps(orig_forward)
        def safe_forward(self, *args, **kwargs):
            # Strip kwargs that the base forward method doesn't accept
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
            return orig_forward(self, *args, **filtered_kwargs)

        cls.forward = safe_forward
        cls._forward_kwargs_patched = True


def ensure_internvl_tokens_configured(model: Any, tokenizer: Any) -> None:
    """Configures special vision-language token IDs (img_context_token_id) on InternVL models."""
    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    if img_context_token_id is None or (isinstance(img_context_token_id, int) and img_context_token_id < 0):
        img_context_token_id = tokenizer.get_vocab().get('<IMG_CONTEXT>', getattr(tokenizer, "eos_token_id", 0))

    # Assign to model and any underlying submodules across PEFT wrappers
    for obj in [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "model", None)
    ]:
        if obj is not None:
            try:
                setattr(obj, "img_context_token_id", img_context_token_id)
            except Exception:
                pass


def load_internvl_model_and_tokenizer(
    model_name_or_path: str,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    is_eval: bool = False
) -> Tuple[Any, Any]:
    """
    Robustly loads InternVL model and tokenizer with GPU acceleration, token setup, and version compatibility.
    """
    cuda_avail = torch.cuda.is_available()

    if device is None:
        device = "cuda" if cuda_avail else "cpu"

    if device_map is None:
        device_map = "auto" if cuda_avail else "cpu"

    if torch_dtype is None:
        if cuda_avail:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

    # Diagnostics
    print("=" * 60)
    print(f"[InternVL Loader] Loading: '{model_name_or_path}'")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[InternVL Loader] Hardware: CUDA GPU ({device_name}) | Total VRAM: {vram_gb:.2f} GB")
        print(f"[InternVL Loader] Precision: {torch_dtype} | device_map: '{device_map}'")
    else:
        print("[InternVL Loader] WARNING: CUDA is NOT available in this PyTorch environment!")
        print("[InternVL Loader] Falling back to CPU. (To enable GPU, reinstall PyTorch with CUDA support).")
        print(f"[InternVL Loader] Precision: {torch_dtype} | device_map: '{device_map}'")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    # Load model with AutoModel (canonical for InternVL) or AutoModelForCausalLM
    try:
        model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code
        )
    except Exception as e:
        print(f"[InternVL Loader] AutoModel fallback to AutoModelForCausalLM due to: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code
        )

    # Ensure model has all_tied_weights_keys attribute if accessed directly
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = getattr(model, "_tied_weights_keys", {})

    # Patch forward method to safely ignore extra kwargs (like inputs_embeds)
    _patch_model_forward(model)

    # Configure img_context_token_id so generate() does not fail assertion
    ensure_internvl_tokens_configured(model, tokenizer)

    if is_eval:
        model = model.eval()

    return model, tokenizer
