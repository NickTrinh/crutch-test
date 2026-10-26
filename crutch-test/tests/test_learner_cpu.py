"""CPU check of the learning student's training path on a tiny model.

Checks, on real dialogs from a learner inputs file:
  1. SGD on the student's own tokens lowers its training loss, more at a larger
     learning rate;
  2. every dialog starts from the same fresh adapter (reset works);
  3. a saved adapter reloads and generates for a test prompt.
The GPU run swaps the tiny model for Llama-3.1-8B and generation for vLLM.

Usage: python tests/test_learner_cpu.py results/learner_inputs_<run>.json.gz
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import learner_core as lc  # noqa: E402

TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = lc.read_inputs(sys.argv[1])
    tok = AutoTokenizer.from_pretrained(TINY)
    model = lc.attach_lora(AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32))
    device = torch.device("cpu")
    dialogs = [d for d in data["dialogs"] if d["arm"] in ("N", "H")][:2]
    examples = [lc.build_example(tok, d["messages"], d["trainable"]) for d in dialogs]

    drops = {}
    for lr in (0.01, 0.3):
        drops[lr] = [lc.train_one(model, ex, lr, seed=0, device=device)["drop"] for ex in examples]
        print(f"lr {lr}: loss drops {[round(x, 3) for x in drops[lr]]}")
    assert all(x > 0 for x in drops[0.3]), "training should lower the student's own loss"
    assert sum(drops[0.3]) > sum(drops[0.01]), "a larger learning rate should learn more"

    lc.reset_lora(model, seed=0)
    first = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    lc.train_one(model, examples[0], 0.3, seed=0, device=device)
    lc.reset_lora(model, seed=0)
    again = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    assert all(torch.equal(first[n], again[n]) for n in first), "reset must restore the fresh adapter"
    print("reset restores the same fresh adapter: ok")

    lc.train_one(model, examples[0], 0.3, seed=0, device=device)
    with tempfile.TemporaryDirectory() as tmp:
        lc.save_adapter(model, tmp)
        base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
        reloaded = PeftModel.from_pretrained(base, tmp).eval()
        test = dialogs[0]["tests"][0]
        ids = tok.apply_chat_template(test["messages"], add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        out = reloaded.generate(ids, max_new_tokens=24, do_sample=False)
        print("adapter reloads and generates:", repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)[:80]))
        print("files:", sorted(p.name for p in Path(tmp).iterdir()))
    print(f"picked lr from these drops (target {lc.TARGET_DROP}): {lc.pick_lr(drops)}")


if __name__ == "__main__":
    main()
