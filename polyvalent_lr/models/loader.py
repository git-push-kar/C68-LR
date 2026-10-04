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
    Wraps model.forward and model.generate to safely route text-only training & inference
    directly to the underlying language_model backbone, eliminating multimodal artifact crashes
    (such as image_flags.squeeze, inputs_embeds mismatch, or pixel_values assertions).
    """
    cls = model.__class__
    if getattr(cls, "_forward_kwargs_patched", False):
        return

    orig_forward = cls.forward
    orig_generate = getattr(cls, "generate", None)

    @functools.wraps(orig_forward)
    def safe_forward(self, *args, **kwargs):
        pixel_values = kwargs.get("pixel_values", None)
        # For pure text inputs (training / inference without images):
        if len(args) == 0 and pixel_values is None and hasattr(self, "language_model"):
            lm = self.language_model
            lm_sig = inspect.signature(lm.forward)
            valid_lm_kwargs = set(lm_sig.parameters.keys())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in lm_sig.parameters.values())
            lm_kwargs = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in valid_lm_kwargs}
            return lm(**lm_kwargs)

        # Fallback to multimodal forward with kwargs filtering
        sig = inspect.signature(orig_forward)
        valid_kwargs = set(sig.parameters.keys())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        filtered_kwargs = kwargs if has_var_kw else {k: v for k, v in kwargs.items() if k in valid_kwargs}
        if "pixel_values" in valid_kwargs and "pixel_values" not in filtered_kwargs and len(args) == 0:
            filtered_kwargs["pixel_values"] = None
        return orig_forward(self, *args, **filtered_kwargs)

    cls.forward = safe_forward

    if orig_generate is not None:
        @functools.wraps(orig_generate)
        def safe_generate(self, *args, **kwargs):
            pixel_values = kwargs.get("pixel_values", None)
            if len(args) == 0 and pixel_values is None and hasattr(self, "language_model") and hasattr(self.language_model, "generate"):
                return self.language_model.generate(*args, **kwargs)
            return orig_generate(self, *args, **kwargs)

        cls.generate = safe_generate

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


def _remove_accelerate_hooks(model: Any) -> None:
    """Removes any accelerate dispatch hooks (AlignDevicesHook) to allow high-throughput GPU training."""
    try:
        from accelerate.hooks import remove_hook_from_module
        for module in model.modules():
            remove_hook_from_module(module, recurse=False)
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

    # For evaluation, device_map="auto" is convenient; for training, device_map=None avoids Accelerate hook overhead
    if device_map is None and is_eval:
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
        print(f"[InternVL Loader] Precision: {torch_dtype} | device_map: {repr(device_map)} | target_device: '{device}'")
    else:
        print("[InternVL Loader] WARNING: CUDA is NOT available in this PyTorch environment!")
        print("[InternVL Loader] Falling back to CPU. (To enable GPU, reinstall PyTorch with CUDA support).")
        print(f"[InternVL Loader] Precision: {torch_dtype} | device_map: {repr(device_map)}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
        model_kwargs["low_cpu_mem_usage"] = True

    # Load model with AutoModel (canonical for InternVL) or AutoModelForCausalLM
    try:
        model = AutoModel.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
    except Exception as e:
        print(f"[InternVL Loader] AutoModel fallback to AutoModelForCausalLM due to: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )

    # If device_map was None and cuda is available, move model directly to target device
    if device_map is None and cuda_avail and device.startswith("cuda"):
        _remove_accelerate_hooks(model)
        model = model.to(device)

    # Ensure model has all_tied_weights_keys attribute if accessed directly
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = getattr(model, "_tied_weights_keys", {})

    # Patch forward and generate to safely route text-only operations to language_model
    _patch_model_forward(model)

    # Configure img_context_token_id so generate() does not fail assertion
    ensure_internvl_tokens_configured(model, tokenizer)

    if is_eval:
        model = model.eval()

    return model, tokenizer
