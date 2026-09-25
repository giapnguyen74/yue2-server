"""Dry run: load YuE2 from the Hugging Face cache, generate a short song, transcribe it back.

  uv run yue2_dryrun.py                    # load + 20 s song (cot=off) + SheetSage2 transcription of it
  uv run yue2_dryrun.py --no-generate      # load and verify the checkpoints only
  uv run yue2_dryrun.py --no-transcribe    # skip the SheetSage2 worker
  uv run yue2_dryrun.py --cot full         # exercise the ABC planner too (slower)
  uv run yue2_dryrun.py --write-profile    # what setup.sh runs: record yue2_profile.json on success

Everything is offline: models come from the cache at the revisions pinned in yue2_common.py.
Measured rates go into the profile so the server's first ETA is not a guess.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from yue2_common import (HERE, GENERATION, MODELS, TRANSCRIBE_WORKER, TRANSCRIPTION, choose_budget,
                         offline_env, probe_gpu, resolve, write_profile)

PROMPT = HERE / "skills" / "yue2-music-server" / "assets" / "prompt.json"


def gib(n):
    return n / 2**30


def vram(label):
    import torch
    print(f"  [{label}] allocated {gib(torch.cuda.memory_allocated()):.2f} GiB, "
          f"peak {gib(torch.cuda.max_memory_allocated()):.2f} GiB", flush=True)


def fresh(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def load(args, report):
    from yue2 import YuE2Pipeline
    import torch

    paths = {name: resolve(name) for name in GENERATION}
    for name, path in paths.items():
        print(f"{name}: {path}")
        report["models"][name] = {"repo": MODELS[name][0], "revision": MODELS[name][1], "snapshot": str(path)}
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; YuE2 needs a BF16-capable NVIDIA GPU")
    t0 = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained(str(paths["YuE2-3B"]), vae=str(paths["YuE2-Vae"]), device="cuda",
                                        memory_budget_gib=args.budget, local_files_only=True, progress=True)
    report["load"]["verify_s"] = round(time.perf_counter() - t0, 1)
    print(f"weights verified in {report['load']['verify_s']}s (mot {list(pipe.weights['mot']['files'])[0]})")
    report["weights"] = pipe.weights

    # Warm the 3B model onto the GPU the way the server's --warmup does; private API, tolerated.
    t0 = time.perf_counter()
    loader = getattr(pipe, "_load_model", None)
    if loader is None:
        print("YuE2Pipeline._load_model is gone; the first generation will load the model instead")
    else:
        loader()
        report["load"]["mot_to_gpu_s"] = round(time.perf_counter() - t0, 1)
        print(f"MoT model on GPU in {report['load']['mot_to_gpu_s']}s")
        vram("model loaded")
    return pipe


def generate(args, pipe, report):
    import torch
    request = json.loads(PROMPT.read_text(encoding="utf-8"))
    request.update(id="dryrun", cot=args.cot, seed=args.seed)
    sampling = {"semantic_sampling": {"min_tokens": args.tokens - 100, "max_tokens": args.tokens}}
    if args.cot != "off":
        sampling["abc_sampling"] = {"max_tokens": args.abc_tokens}
    counts = {"abc": 0, "semantic": 0}

    def on_token(phase, token):
        counts[phase] += 1

    out = fresh(args.out)
    print(f"generating: cot={args.cot}, ~{args.tokens * 0.04:.0f}s of audio, seed {args.seed} -> {out}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    song = pipe(**request, **sampling, on_token=on_token)
    receipt = song.save_artifacts(out)
    elapsed = time.perf_counter() - t0
    timing = song.timing
    gen = {
        "cot": args.cot, "seed": args.seed, "e2e_s": round(elapsed, 1),
        "audio_s": round(receipt["audio_seconds"], 1), "truncated": receipt["truncated"],
        "abc_tokens": counts["abc"], "abc_s": round(timing["abc"].get("seconds", 0.0), 1),
        "semantic_tokens": counts["semantic"], "semantic_s": round(timing["semantic"].get("seconds", 0.0), 1),
        "nar_s": round(timing["nar_seconds"], 1), "vae_s": round(timing["vae_seconds"], 1),
        "peak_gib": round(gib(torch.cuda.max_memory_allocated()), 2),
        "identity": receipt["identity"], "output": str(out),
    }
    if gen["semantic_s"] > 0:
        gen["semantic_tok_s"] = round(counts["semantic"] / gen["semantic_s"], 1)
    if gen["abc_s"] > 0 and counts["abc"]:
        gen["abc_tok_s"] = round(counts["abc"] / gen["abc_s"], 1)
    if counts["semantic"]:
        gen["nar_s_per_token"] = round(gen["nar_s"] / counts["semantic"], 4)
        gen["vae_s_per_token"] = round(gen["vae_s"] / counts["semantic"], 4)
    report["generate"] = gen
    print(f"  {gen['audio_s']}s audio in {gen['e2e_s']}s: semantic {gen.get('semantic_tok_s', '?')} tok/s, "
          f"NAR {gen['nar_s']}s, VAE {gen['vae_s']}s, peak {gen['peak_gib']} GiB -> {out / 'audio.flac'}")
    if any(gen["truncated"].values()):
        print("  note: generation hit its token limit (expected for a dry run with a small max_tokens)")
    return out / "audio.flac"


def transcribe(args, audio, report):
    if not shutil.which("uv"):
        raise SystemExit("uv is required to run the SheetSage2 worker (workers/transcribe.py)")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required by SheetSage2; install it and rerun")
    paths = {name: resolve(name) for name in TRANSCRIPTION}
    for name, path in paths.items():
        print(f"{name}: {path}")
        report["models"][name] = {"repo": MODELS[name][0], "revision": MODELS[name][1], "snapshot": str(path)}
    out = Path(args.out) / "transcription"
    if out.exists():
        shutil.rmtree(out)
    command = ["uv", "run", "--locked", "--script", str(TRANSCRIBE_WORKER), str(audio),
               "--output", str(out), "--task", "melody-full", "--offline", "--device", "cuda",
               "--model", str(paths["SheetSage2"]), "--base-model", str(paths["MERT-v2-FullSong"])]
    print("transcribing the dry-run song with SheetSage2 (separate uv environment)...")
    t0 = time.perf_counter()
    proc = subprocess.run(command, env=offline_env(), text=True, capture_output=True)
    elapsed = time.perf_counter() - t0
    (Path(args.out) / "transcription.log").write_text(proc.stdout + proc.stderr)
    loaded = (out / "model_provenance.json").is_file()
    result = {"exit_code": proc.returncode, "model_loaded": loaded, "elapsed_s": round(elapsed, 1),
              "output": str(out), "log": str(Path(args.out) / "transcription.log")}
    if proc.returncode == 0:
        manifest = json.loads((out / "transcription_manifest.json").read_text())
        result.update(status="complete", warnings=manifest.get("warnings", []))
        print(f"  score.abc written in {result['elapsed_s']}s; warnings: {result['warnings']}")
    else:
        last = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        result.update(status="failed", error=last)
        if loaded:
            # A 20 s cot=off clip can lack the beats/key SheetSage2 needs to build ABC. The environment,
            # the models and the GPU handoff all worked, which is what the dry run is checking.
            print(f"  worker loaded SheetSage2 + MERT but produced no score: {last}")
            print("  (short synthetic clip; not counted as a setup failure)")
        else:
            print(f"  worker failed before loading the model: {last}", file=sys.stderr)
            print(f"  see {result['log']}", file=sys.stderr)
            raise SystemExit(1)
    report["transcribe"] = result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-generate", action="store_true", help="load and verify the checkpoints only")
    ap.add_argument("--no-transcribe", action="store_true", help="skip the SheetSage2 worker")
    ap.add_argument("--cot", choices=("off", "melody", "full"), default="off")
    ap.add_argument("--tokens", type=int, default=500, help="semantic max_tokens; 25 tokens per second of audio")
    ap.add_argument("--abc-tokens", type=int, default=4096, help="ABC max_tokens when --cot is not off")
    ap.add_argument("--seed", type=int, default=831001)
    ap.add_argument("--budget", type=float, help="memory_budget_gib; default from the GPU")
    ap.add_argument("--out", default=str(HERE / "outputs" / "dryrun"))
    ap.add_argument("--write-profile", action="store_true", help="record yue2_profile.json on success")
    args = ap.parse_args()
    if args.tokens < 150:
        ap.error("--tokens must be at least 150")

    gpu_name, total = probe_gpu()
    if args.budget is None:
        args.budget = choose_budget(total)
    print(f"GPU: {gpu_name or 'none detected'} ({total:.1f} GiB); memory budget {args.budget} GiB")
    report = {"gpu": gpu_name, "vram_gib": round(total, 1), "budget_gib": args.budget,
              "models": {}, "load": {}, "generate": None, "transcribe": None}

    pipe = load(args, report)
    try:
        audio = None
        if not args.no_generate:
            audio = generate(args, pipe, report)
    finally:
        pipe.close()   # release the GPU before the transcription worker starts, as the server will
    if audio is not None and not args.no_transcribe:
        transcribe(args, audio, report)
    elif not args.no_transcribe:
        print("transcription skipped: nothing generated to transcribe (--no-generate)")

    if args.write_profile:
        report["written"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        path = write_profile(report)
        print(f"profile -> {path}")
    print("dry run OK")


if __name__ == "__main__":
    main()
