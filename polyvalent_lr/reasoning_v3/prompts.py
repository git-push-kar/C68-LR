"""Clean, leakage-free prompt/target templates for v3.

The v2 model leaked scaffold tokens (``reasoning``/``<br />``/``answer``) and raw
HTML. v3 uses a single, explicit, markup-free contract:

    <reasoning> ...step-by-step... </reasoning>
    <answer> ...final answer... </answer>

The chat wrapper is the tokenizer's native ChatML (``<|im_start|>...``) so the
assistant EOS (``<|im_end|>``) is the supervised terminator.
"""
from __future__ import annotations

SYSTEM_REASONING = (
    "You are a rigorous logical reasoning engine. Work through the problem "
    "step by step inside <reasoning>...</reasoning>, then give the final answer "
    "inside <answer>...</answer>. In the reasoning, update what you know about "
    "each desk after every clue. Keep the final answer concise and complete."
)

SYSTEM_LOGIC = (
    "You are a rigorous logical reasoning engine. Work through the problem "
    "step by step inside <reasoning>...</reasoning>, then give the final answer "
    "inside <answer>...</answer>. State only conclusions that follow from the "
    "given premises."
)

SYSTEM_CHAT = "You are a helpful, knowledgeable AI assistant."

REASONING_OPEN = "<reasoning>"
REASONING_CLOSE = "</reasoning>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_chat_prompt(system: str, user: str) -> str:
    """ChatML prompt ending at the assistant header (to be continued by target)."""
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{user}{IM_END}\n"
        f"{IM_START}assistant\n"
    )


def build_reasoning_target(reasoning: str, answer: str) -> str:
    return (
        f"{REASONING_OPEN}\n{reasoning.strip()}\n{REASONING_CLOSE}\n"
        f"{ANSWER_OPEN}\n{answer.strip()}\n{ANSWER_CLOSE}"
    )


def build_chat_target(assistant_text: str) -> str:
    """Plain assistant reply for rehearsal data (no scaffolding tags)."""
    return assistant_text.strip()


def build_puzzle_prompt(problem_text: str) -> str:
    return build_chat_prompt(SYSTEM_REASONING, problem_text)


def build_puzzle_target(trace_text: str, table_text: str) -> str:
    return build_reasoning_target(trace_text, table_text)


def inference_system_prompt() -> str:
    """System prompt used by the Stage-4 UIs (matches the training trace format)."""
    return SYSTEM_REASONING
