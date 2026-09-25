"""Stand-in for workers/lyrics.py: same CLI and files, no model. FAKE_WORKER_FAIL=load|asr to fail."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lyrics-file", type=Path)
    ap.add_argument("--language", default="auto")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not args.audio.is_file():
        print("no such audio", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=False)
    mode = os.environ.get("FAKE_WORKER_FAIL", "")
    if mode == "load":
        print("RuntimeError: could not load Qwen3-ASR", file=sys.stderr)
        return 2
    (args.output / "model_provenance.json").write_text(json.dumps({"model": args.model, "fake": True}))
    time.sleep(float(os.environ.get("FAKE_WORKER_SLEEP", "0.1")))
    if mode == "asr":
        (args.output / "failure.json").write_text(json.dumps({"status": "failed", "error": "decode exploded"}))
        return 2
    reference = args.lyrics_file.read_text(encoding="utf-8") if args.lyrics_file else None
    text = "neon fades along the lane footsteps keep the time of rain"
    passes = []
    for i in range(args.passes):
        entry = {"pass": i + 1, "text": text, "language": "English", "normalized": text, "seconds": 0.1}
        if reference is not None:
            entry.update(unit_error_rate=0.25 - 0.05 * i, per=0.12 - 0.02 * i, hyp_phonemes=40)
        passes.append(entry)
    best = passes[-1] if reference is not None else passes[0]
    (args.output / "transcript.txt").write_text(best["text"] + "\n")
    report = {"status": "complete", "audio_seconds": 1.0, "language_requested": args.language,
              "language_detected": "English", "passes": passes, "best_pass": best["pass"],
              "unit_error_rate": best.get("unit_error_rate"), "per": best.get("per"),
              "reference": None if reference is None else {"normalized": reference.lower(), "units": 12, "phonemes": 40,
                                                            "unit_kind": "words"},
              "scoring": "fake"}
    (args.output / "lyrics_asr.json").write_text(json.dumps(report))
    artifacts = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "lyrics_manifest.json").write_text(json.dumps({
        "status": "complete", "source_audio_sha256": sha(args.audio), "artifacts": artifacts}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
