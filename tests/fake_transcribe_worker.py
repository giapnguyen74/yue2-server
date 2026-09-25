"""Stand-in for workers/transcribe.py: same CLI, writes the same files, no models.

Set FAKE_WORKER_FAIL=load to exit before writing model_provenance.json, or =abc to fail after loading.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ABC = (Path(__file__).with_name("fixtures") / "melody.abc").read_text(encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--task", default="full")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model")
    ap.add_argument("--base-model")
    args = ap.parse_args()
    if not args.audio.is_file():
        print("no such audio", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=False)
    mode = os.environ.get("FAKE_WORKER_FAIL", "")
    if mode == "load":
        print("RuntimeError: could not load SheetSage2", file=sys.stderr)
        return 2
    (args.output / "model_provenance.json").write_text(json.dumps({"model": args.model, "fake": True}))
    time.sleep(float(os.environ.get("FAKE_WORKER_SLEEP", "0.2")))
    if mode == "abc":
        (args.output / "failure.json").write_text(json.dumps({"status": "failed", "error": "no usable ABC"}))
        print("ValueError: Transcription produced no usable ABC", file=sys.stderr)
        return 2
    (args.output / "score.abc").write_text(ABC)
    (args.output / "melody.mid").write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60MTrk\x00\x00\x00\x04\x00\xff\x2f\x00")
    (args.output / "events.json").write_text("[]")
    artifacts = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "transcription_manifest.json").write_text(json.dumps({
        "status": "complete", "warnings": ["fake"], "source_audio_sha256": sha(args.audio), "artifacts": artifacts}))
    print(f"Saved {args.output / 'score.abc'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
