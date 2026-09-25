#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "qwen-asr==0.0.6",        # pins transformers 4.57.6, accelerate 1.12.0 (conflicts with yue2-infer's 1.13.0)
#     "torch==2.10.0",          # cu128 wheels: the same build the server uses on this GPU
#     "g2p-en==2.1.0",          # English grapheme-to-phoneme (ARPAbet), needs nltk data (setup.sh fetches it)
#     "pypinyin==0.55.0",       # Mandarin initials/finals
#     "soundfile==0.13.1",
# ]
# ///
"""Recognise sung lyrics with Qwen3-ASR and score them against reference lyrics (WER/CER and PER).

  lyrics.py <audio> --output <dir> --model <Qwen3-ASR snapshot> [--lyrics-file ref.txt]
            [--language English|Chinese|auto] [--passes N] [--offline] [--device cuda]

Runs as a subprocess of yue2-server in its own uv environment. Writes into <dir>:
  model_provenance.json   after the model is loaded (the server's phase marker)
  transcript.txt          best pass (lowest PER against the reference, else the first pass)
  lyrics_asr.json         every pass, detected language, normalized reference, WER/CER and PER
  lyrics_manifest.json    {"status": "complete", "artifacts": {name: sha256}}
  failure.json            on error

Scoring is deliberate and documented rather than a claim to match WildSongBench's undisclosed evaluator:
text is lower-cased, section tags such as [Verse] and all punctuation are dropped; CJK characters are one
unit each and other words are whitespace units (so "unit error rate" is CER for Chinese, WER for English);
phonemes are ARPAbet without stress from g2p_en for Latin-script words and tone-less pinyin
initial + final from pypinyin for CJK characters. PER = Levenshtein(ref, hyp) / len(ref phonemes).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
SECTION = re.compile(r"\[[^\]\n]{0,40}\]")
LANGUAGES = {"auto": None, "english": "English", "chinese": "Chinese"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


# ── text ──────────────────────────────────────────────────────────────────────

def normalize(text):
    text = SECTION.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).lower()
    out = []
    for ch in text:
        if CJK.match(ch):
            out.append(f" {ch} ")
        elif ch.isalnum() or ch in "'’":
            out.append("'" if ch == "’" else ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def units(normalized):
    return normalized.split()


class Phonemizer:
    def __init__(self):
        self._g2p = None
        self._pinyin = None

    def _english(self, word):
        if self._g2p is None:
            from g2p_en import G2p
            self._g2p = G2p()
        return [re.sub(r"\d", "", p) for p in self._g2p(word) if p.strip() and p != " "]

    def _chinese(self, char):
        if self._pinyin is None:
            from pypinyin import Style, pinyin
            self._pinyin = (pinyin, Style)
        pinyin, Style = self._pinyin
        initial = pinyin(char, style=Style.INITIALS, strict=False)[0][0]
        final = pinyin(char, style=Style.FINALS, strict=False)[0][0]
        return [p for p in (initial, final) if p]

    def __call__(self, normalized):
        phones = []
        for unit in units(normalized):
            if CJK.match(unit):
                phones += self._chinese(unit)
            elif unit.isdigit():
                phones += self._english(unit)
            else:
                phones += self._english(unit)
        return phones


def levenshtein(a, b):
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def rate(ref, hyp):
    return levenshtein(ref, hyp) / len(ref) if ref else None


# ── audio and model ───────────────────────────────────────────────────────────

def load_audio(path):
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.mean(axis=1)), sr, len(data) / sr


def run(args):
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    language_key = args.language.lower()
    if language_key not in LANGUAGES:
        raise ValueError("--language must be auto, English or Chinese")
    language = LANGUAGES[language_key]
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    reference = args.lyrics_file.read_text(encoding="utf-8") if args.lyrics_file else None
    write_json(out / "input.json", {
        "source_name": args.audio.name, "source_audio_sha256": sha256(args.audio),
        "model": args.model, "language": args.language, "passes": args.passes,
        "reference_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest() if reference else None,
    })
    try:
        import torch
        from qwen_asr import Qwen3ASRModel

        waveform, sr, seconds = load_audio(args.audio)
        model = Qwen3ASRModel.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=args.device,
            max_inference_batch_size=1, max_new_tokens=args.max_new_tokens,
        )
        snapshot = Path(args.model)
        write_json(out / "model_provenance.json", {
            "model": args.model,
            "snapshot_sha256": {p.name: sha256(p) for p in sorted(snapshot.iterdir())
                                if p.is_file() and p.suffix in {".json", ".safetensors"}} if snapshot.is_dir() else {},
            "packages": {name: importlib.metadata.version(name) for name in ("qwen-asr", "torch", "transformers")},
            "device": args.device, "dtype": "bfloat16",
        })
        phonemize = Phonemizer()
        ref_norm = normalize(reference) if reference else None
        ref_units = units(ref_norm) if ref_norm else None
        ref_phones = phonemize(ref_norm) if ref_norm else None

        passes = []
        for index in range(args.passes):
            start = time.perf_counter()
            result = model.transcribe(audio=(waveform, sr), language=language)[0]
            hyp_norm = normalize(result.text or "")
            entry = {"pass": index + 1, "text": result.text, "language": getattr(result, "language", None),
                     "normalized": hyp_norm, "seconds": round(time.perf_counter() - start, 2)}
            if ref_norm is not None:
                hyp_phones = phonemize(hyp_norm)
                entry["unit_error_rate"] = rate(ref_units, units(hyp_norm))
                entry["per"] = rate(ref_phones, hyp_phones)
                entry["hyp_phonemes"] = len(hyp_phones)
            passes.append(entry)
            print(f"pass {index + 1}: {entry.get('language')} per={entry.get('per')} uer={entry.get('unit_error_rate')}",
                  file=sys.stderr, flush=True)
        best = min(passes, key=lambda e: e["per"]) if ref_norm is not None else passes[0]
        (out / "transcript.txt").write_text((best["text"] or "") + "\n", encoding="utf-8")
        report = {
            "status": "complete", "audio_seconds": round(seconds, 2), "language_requested": args.language,
            "language_detected": best.get("language"), "passes": passes, "best_pass": best["pass"],
            "unit_error_rate": best.get("unit_error_rate"), "per": best.get("per"),
            "reference": None if ref_norm is None else {
                "normalized": ref_norm, "units": len(ref_units), "phonemes": len(ref_phones),
                "unit_kind": "characters" if any(CJK.match(u) for u in ref_units) else "words"},
            "scoring": "lowercase; [section] tags and punctuation dropped; CJK chars are units, other words are units; "
                       "PER over ARPAbet (no stress) for Latin words and tone-less pinyin initial+final for CJK; "
                       "best pass = lowest PER. Not the WildSongBench evaluator.",
        }
        write_json(out / "lyrics_asr.json", report)
        write_json(out / "lyrics_manifest.json", {
            "status": "complete", "source_audio_sha256": sha256(args.audio),
            "artifacts": {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()},
        })
        print(f"Saved {out / 'lyrics_asr.json'}; per={report['per']} uer={report['unit_error_rate']}")
    except Exception as exc:
        write_json(out / "failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
        raise


def prepare():
    """Fetch the nltk resources g2p_en looks up at import (setup time; the worker itself runs offline)."""
    import nltk
    for resource in ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"):
        nltk.download(resource, quiet=True)
    from g2p_en import G2p
    print("g2p_en ready:", " ".join(G2p()("lyrics")))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, help="Fresh output directory")
    parser.add_argument("--model", help="Qwen3-ASR snapshot directory or repo id")
    parser.add_argument("--prepare", action="store_true", help="download the nltk data g2p_en needs, then exit")
    parser.add_argument("--lyrics-file", type=Path, help="Reference lyrics; without it only a transcript is produced")
    parser.add_argument("--language", default="auto", help="auto, English or Chinese")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prepare:
        return prepare()
    if args.audio is None or args.output is None or args.model is None:
        parser.error("audio, --output and --model are required")
    if args.passes < 1:
        parser.error("--passes must be at least 1")
    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
