"""Run the learning student (view L) on a Modal A100. Run it from your own computer:

    pip install modal
    python -m modal setup                      # once: links your Modal account
    modal run crutch-test/learner_modal.py --run pilot50v2 --limit 2   # 15-minute check
    modal deploy crutch-test/learner_modal.py                         # then the real run:
    python crutch-test/learner_spawn.py start --run pilot50v2         # see learner_spawn.py

The real run is spawned on the deployed app so it survives this computer
dropping its connection or sleeping; an ephemeral `modal run` dies with it.

It reads results/learner_inputs_<run>.json.gz and makes two GPU calls:
  1. train_adapters: picks the learning rate on the N and H dialogs only
     (learner_core.pick_lr), then trains one LoRA adapter per dialog on the
     student's own tokens and saves it to a Modal volume;
  2. sample_tests: in a fresh container, so vLLM starts on a clean GPU,
     samples every test with each adapter and with the frozen model.
It writes results/learner_outputs_<run>.json.gz: commit and push that file.
--limit N uses only the first N dialogs per arm and writes
learner_outputs_<run>_limitN.json.gz, which needs no commit: it only checks
the GPU path before the hour-long run. Grading happens in the cloud session. Expect about an hour and $3-5 of
Modal credit (an A100-80GB, priced per second); the first run also
downloads the 16 GB model into the volume, which later runs reuse.
"""

import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so learner_core can be shipped to the container

# torch 2.13 ships the CUDA 13 runtime; Modal hosts run driver 580 (CUDA 13.0).
# transformers and peft are the versions tests/test_learner_cpu.py passed with.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0", "torch==2.13.0", "transformers==5.19.0",
                    "peft==0.19.1", "accelerate==1.15.0")
    # Plain temperature sampling needs no FlashInfer kernels; this keeps vLLM off them.
    .env({"HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("learner_core")
)
cache = modal.Volume.from_name("crutch-test-cache", create_if_missing=True)
app = modal.App("crutch-test-learner", image=image)


@app.function(gpu="A100-80GB", timeout=3 * 60 * 60, volumes={"/cache": cache})
def train_adapters(inputs: bytes) -> dict:
    import gzip
    import json
    import os
    import time

    # Set before torch loads. The Oct 26 run died with 9.75 GiB reserved but unusable;
    # expandable segments cut that fragmentation. Training container only (vLLM untouched).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import learner_core as lc

    t0 = time.time()
    data = json.loads(gzip.decompress(inputs))
    name, rev = data["model"]["name"], data["model"]["revision"]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    base = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map="cuda")
    cache.commit()  # keep the downloaded weights for later runs
    # The longest dialogs (~5.5K tokens) ran out of memory on the 80 GB A100 when every
    # layer kept its activations for backward. Checkpointing recomputes them instead:
    # roughly 30% slower, same gradients.
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()  # gradients reach the LoRA weights even if checkpointing is reentrant
    model = lc.attach_lora(base)
    device = torch.device("cuda")
    dialogs = data["dialogs"]
    examples = [lc.build_example(tok, d["messages"], d["trainable"]) for d in dialogs]

    # The learning rate comes from N and H dialogs only; T never informs it.
    # A fixed rate (settings.force_lr) skips the pick: the robustness rerun needs
    # it because lr 0.03 and 0.1 sit almost equally far from the 25% target.
    calibration = {}
    lr = data["settings"].get("force_lr")
    if lr is None:
        calibration = {lr: [] for lr in lc.LR_GRID}
        for lr in lc.LR_GRID:
            for i in lc.calibration_ids(data):
                calibration[lr].append(lc.train_one(model, examples[i], lr, seed=i, device=device)["drop"])
            drops = sorted(calibration[lr])
            print(f"[learner] lr {lr}: median loss drop {drops[len(drops) // 2]:.3f}", flush=True)
        lr = lc.pick_lr(calibration)
    print(f"[learner] {'fixed' if calibration == {} else 'picked'} lr {lr}", flush=True)

    # One adapter per dialog, each from the same fresh start.
    out_dir = Path("/cache/adapters") / data["run"]
    train = []
    for i, d in enumerate(dialogs):
        stats = lc.train_one(model, examples[i], lr, seed=i, device=device)
        lc.save_adapter(model, out_dir / str(i))
        train.append({"arm": d["arm"], "key": d["key"], **stats})
    cache.commit()
    minutes = (time.time() - t0) / 60
    print(f"[learner] trained {len(train)} adapters in {minutes:.1f} min", flush=True)
    return {"lr": lr, "calibration": {str(k): v for k, v in calibration.items()},
            "train": train, "train_minutes": minutes}


@app.function(gpu="A100-80GB", timeout=3 * 60 * 60, volumes={"/cache": cache})
def sample_tests(inputs: bytes) -> dict:
    import gzip
    import json
    import shutil
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    import learner_core as lc

    t0 = time.time()
    cache.reload()
    data = json.loads(gzip.decompress(inputs))
    name, rev = data["model"]["name"], data["model"]["revision"]
    adapters = Path("/cache/adapters") / data["run"]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    llm = LLM(model=name, revision=rev, dtype="bfloat16", seed=0, enable_lora=True,
              max_lora_rank=lc.LORA["r"], max_loras=8, max_cpu_loras=32,
              max_model_len=8192, gpu_memory_utilization=0.85)

    def prompt(messages):
        out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return {"prompt_token_ids": list(out["input_ids"] if hasattr(out, "keys") else out)}

    answers = []
    requests = lc.test_requests(data)
    for frozen in (True, False):
        batch = [r for r in requests if (r["adapter"] == "base") == frozen]
        params = [SamplingParams(n=r["n"], temperature=data["settings"]["temperature"],
                                 max_tokens=data["settings"]["max_tokens"]) for r in batch]
        loras = None if frozen else [
            LoRARequest(f"d{r['adapter']}", r["adapter"] + 1, str(adapters / str(r["adapter"]))) for r in batch]
        outputs = llm.generate([prompt(r["messages"]) for r in batch], params, lora_request=loras)
        for r, o in zip(batch, outputs):
            answers.append({"adapter": r["adapter"], "arm": r["arm"], "key": r["key"], "kind": r["kind"],
                            "answer": r["answer"], "texts": [c.text for c in o.outputs],
                            "finish": [c.finish_reason for c in o.outputs],
                            "tokens": [len(c.token_ids) for c in o.outputs]})
        print(f"[learner] sampled {'frozen' if frozen else 'adapted'} tests: {len(batch)} prompts", flush=True)
    shutil.rmtree(adapters, ignore_errors=True)  # about 13 GB; the outputs carry what is needed
    cache.commit()
    return {"answers": answers, "sample_minutes": (time.time() - t0) / 60}


@app.function(timeout=6 * 60 * 60, volumes={"/cache": cache})
def pipeline(inputs: bytes) -> dict:
    """Train, then sample, both on Modal, and keep the result in the volume.

    Chaining the two calls remotely lets a detached run (`modal run --detach`) finish
    even if the local client goes away: on Oct 26 the client died twice mid-run (a
    session restart, then a network drop), and each time Modal stopped the app.
    """
    import gzip
    import json

    run = json.loads(gzip.decompress(inputs))["run"]
    trained = train_adapters.remote(inputs)
    sampled = sample_tests.remote(inputs)
    out = {"run": run, **trained, **sampled}
    path = Path("/cache/outputs") / f"learner_outputs_{run}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(out).encode()))
    cache.commit()
    print(f"[learner] saved {path} in the volume", flush=True)
    return out


def load_inputs(run: str, limit: int = 0, lr: float | None = None) -> bytes:
    """The run's learner inputs as gzipped JSON; shared with learner_spawn.py.

    limit > 0 keeps the first `limit` dialogs of each arm (the quick GPU check)
    and renames the run <run>_limit<N>, so its outputs never overwrite a real run's.
    lr fixes the learning rate instead of picking it, and adds _lr<lr> to the name.
    """
    import collections
    import gzip
    import json

    src = HERE / "results" / f"learner_inputs_{run}.json.gz"
    if not src.exists():
        raise SystemExit(f"{src} is missing: pull the branch first")
    data = json.loads(gzip.decompress(src.read_bytes()))
    if limit:
        keep, seen = [], collections.Counter()
        for d in data["dialogs"]:
            seen[d["arm"]] += 1
            if seen[d["arm"]] <= limit:
                keep.append(d)
        data["dialogs"] = keep
        data["run"] = f"{run}_limit{limit}"
    if lr is not None:
        data["settings"]["force_lr"] = lr
        data["run"] = f"{data['run']}_lr{lr:g}"
    return gzip.compress(json.dumps(data).encode())


def write_outputs(out: dict) -> Path:
    """Write results/learner_outputs_<run>.json.gz and print a one-line summary."""
    import gzip
    import json

    name = out["run"]
    dst = HERE / "results" / f"learner_outputs_{name}.json.gz"
    dst.write_bytes(gzip.compress(json.dumps(out).encode()))
    answers = [t for a in out["answers"] for t in a["finish"]]
    print(f"lr {out['lr']}; {len(out['train'])} adapters in {out['train_minutes']:.0f} min; "
          f"{len(answers)} test answers in {out['sample_minutes']:.0f} min, "
          f"{sum(f == 'length' for f in answers) / max(1, len(answers)):.0%} cut off at the cap")
    if "_limit" in name:
        print(f"Wrote {dst}. The GPU path works; no need to commit this file. Now run without --limit.")
    else:
        print(f"Wrote {dst}. Commit and push it:")
        print(f"  git add {dst.relative_to(HERE.parent)} && git commit -m \"Learning student outputs ({name})\" && git push")
    return dst


@app.local_entrypoint()
def main(run: str = "pilot50v2", limit: int = 0):
    # Train and sample on Modal; the result is also saved in the volume.
    write_outputs(pipeline.remote(load_inputs(run, limit)))
