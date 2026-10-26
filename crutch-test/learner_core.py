"""The learning student (view L): learn from your own practice, then take the test.

For each dialog, a LoRA adapter on the student model trains on the tokens the
student itself wrote. The tutor's turns are context with no loss, and the
scripted opener is excluded. Training is plain SGD on the summed token loss,
so a student who wrote more learns more; Adam would rescale every update to
the same size. The adapted student then takes the unaided test with no
transcript in view, so what it learned has to be in its weights.

The learning rate is fixed before any test: on the N and H dialogs only, it is
the grid value whose median drop in training loss after STEPS steps is
closest to TARGET_DROP. T dialogs never inform the setting.

Shared by learner_modal.py (GPU, vLLM) and test_learner_cpu.py (CPU, tiny model).
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
from pathlib import Path

STEPS = 4
LOSS_SCALE = 1000.0  # summed token loss / 1,000: the update grows with the student's own tokens
LR_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)  # SGD on LoRA needs large rates; the tiny CPU model dropped 8% at 0.3
TARGET_DROP = 0.25
CALIBRATION_DIALOGS = 20  # per calibration arm, the first by problem key: enough for a median, half the GPU time
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
MAX_TRAIN_TOKENS = 8192


# --- inputs --------------------------------------------------------------------

def read_inputs(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def label_spans(rendered: str, messages: list[dict], trainable: list[bool]) -> list[tuple[int, int]]:
    """Character spans, in the rendered chat text, of the trainable messages' content.

    Messages are found in order, so a content string that also appears earlier
    (say, a quoted line) cannot be matched to the wrong turn.
    """
    spans, cursor = [], 0
    for msg, train in zip(messages, trainable):
        content = msg["content"]
        start = rendered.find(content, cursor)
        if start < 0:
            raise ValueError(f"content not found in the rendered chat: {content[:60]!r}")
        end = start + len(content)
        if train:
            spans.append((start, end))
        cursor = end
    return spans


def build_example(tokenizer, messages: list[dict], trainable: list[bool]) -> dict:
    """Token ids plus labels: -100 everywhere except the student's own tokens."""
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    spans = label_spans(rendered, messages, trainable)
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True,
                    truncation=True, max_length=MAX_TRAIN_TOKENS)
    labels = []
    for tok_id, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        inside = any(s <= a and b <= e and b > a for s, e in spans)
        labels.append(tok_id if inside else -100)
    return {"input_ids": enc["input_ids"], "labels": labels,
            "n_train_tokens": sum(1 for x in labels if x != -100)}


# --- training --------------------------------------------------------------------

def attach_lora(model):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))


def reset_lora(model, seed: int) -> None:
    import torch
    from peft.tuners.lora import LoraLayer
    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.reset_lora_parameters("default", True)


def summed_loss(model, example: dict, device):
    import torch
    ids = torch.tensor([example["input_ids"]], device=device)
    labels = torch.tensor([example["labels"]], device=device)
    logits = model(input_ids=ids).logits[:, :-1].float()
    target = labels[:, 1:]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100, reduction="sum")


def train_one(model, example: dict, lr: float, seed: int, device) -> dict:
    """Reset the adapter, take STEPS SGD steps on the student's own tokens."""
    import torch
    reset_lora(model, seed)
    n = max(1, example["n_train_tokens"])
    if example["n_train_tokens"] == 0:
        return {"n_train_tokens": 0, "loss_before": None, "loss_after": None, "drop": 0.0}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    model.train()
    with torch.no_grad():
        before = summed_loss(model, example, device).item() / n
    for _ in range(STEPS):
        loss = summed_loss(model, example, device) / LOSS_SCALE
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        after = summed_loss(model, example, device).item() / n
    model.eval()
    drop = 0.0 if not math.isfinite(after) else 1.0 - after / before
    return {"n_train_tokens": example["n_train_tokens"], "loss_before": before,
            "loss_after": after, "drop": drop}


def save_adapter(model, path: str | Path) -> None:
    """Save the adapter in PEFT format, weights in bf16 (half the disk; vLLM serves bf16)."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to(torch.bfloat16).contiguous().cpu()
             for k, v in get_peft_model_state_dict(model).items()}
    save_file(state, str(path / "adapter_model.safetensors"))
    model.peft_config["default"].save_pretrained(str(path))


def calibration_ids(data: dict) -> list[int]:
    """Dialogs that set the learning rate: the first CALIBRATION_DIALOGS of each calibration arm."""
    ids = []
    for arm in data["settings"]["calibrate_on"]:
        ids += [i for i, d in enumerate(data["dialogs"]) if d["arm"] == arm][:CALIBRATION_DIALOGS]
    return ids


def pick_lr(calibration: dict[float, list[float]]) -> float:
    """The grid value whose median loss drop on N and H dialogs is closest to TARGET_DROP."""
    medians = {lr: statistics.median(d) for lr, d in calibration.items() if d}
    return min(medians, key=lambda lr: (abs(medians[lr] - TARGET_DROP), -lr))


# --- tests -----------------------------------------------------------------------

def test_requests(data: dict) -> list[dict]:
    """Every (adapter, test) pair to sample: each dialog's adapter plus the frozen base."""
    reqs = []
    for i, d in enumerate(data["dialogs"]):
        for t in d["tests"]:
            reqs.append({"adapter": i, "arm": d["arm"], "key": d["key"], **t})
    seen = set()
    for d in data["dialogs"]:
        for t in d["tests"]:
            k = (d["key"], t["kind"])
            if k not in seen:
                seen.add(k)
                reqs.append({"adapter": "base", "arm": "base", "key": d["key"], **t})
    return reqs


def write_outputs(path: str | Path, payload: dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
