"""Shared by setup.sh, yue2_dryrun.py and the server: pinned model snapshots, the profile, GPU sizing.

The pins are the commits the YuE repository's uv skill records. Every model lives in the Hugging Face
cache (HF_HOME / HF_HUB_CACHE respected) and is resolved offline; nothing here downloads.

  python3 yue2_common.py files YuE2-3B        # "<repo> <revision> <file>..." for `hf download` (stdlib only)
  python3 yue2_common.py names [--transcribe] [--lyrics] [--legacy]
  python3 yue2_common.py cached YuE2-3B       # exit 0 when the pinned snapshot is complete in the cache
  python3 yue2_common.py import <dir> [names] # hard-link <dir>/<name>/ snapshots (hf --local-dir layout) into the cache
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = HERE / "yue2_profile.json"

_YUE2_LICENSES = ["LICENSE", "THIRD_PARTY_NOTICES.md",
                  "licenses/SnakeBeta-NVIDIA-MIT.txt", "licenses/stable-audio-tools-MIT.txt"]
_VAE_FILES = ["config.json", "weights_manifest.json", "model.safetensors", "modeling_vae.py"] + _YUE2_LICENSES

# name -> (repo, revision, files, approximate GB)
MODELS = {
    "YuE2-3B": ("m-a-p/YuE2-3B", "1a96eca688d6ae5d7f0feb88573fec89920fcd19",
                ["config.json", "generation_config.json", "yue2_generation_config.json", "weights_manifest.json",
                 "model.safetensors", "qwen.tiktoken", "modeling_yue2.py"] + _YUE2_LICENSES, 7.3),
    "YuE2-Vae": ("m-a-p/YuE2-Vae", "95535e72a97bc0f09b8ada125d26b4009428c0e8", _VAE_FILES, 0.5),
    # Benchmark-protocol decoder; optional. The Hub's main on 2026-09-25 (no pin in the YuE skills).
    "YuE2-Vae-legacy": ("m-a-p/YuE2-Vae-legacy", "5ddd12f79acb90d24b3a672dcd2ebf88da7c92a9", _VAE_FILES, 0.5),
    "SheetSage2": ("m-a-p/SheetSage2", "eab522a8168e8b8b8c4856bf8609cd86198f01fe",
                   ["__init__.py", "audio_sheetsage2.py", "configuration_mert2.py", "configuration_sheetsage2.py",
                    "durations_sheetsage2.py", "exports_sheetsage2.py", "generation_sheetsage2.py",
                    "io_sheetsage2.py", "labels_sheetsage2.py", "midi_sheetsage2.py", "modeling_mert2.py",
                    "modeling_sheetsage2.py", "notation_sheetsage2.py", "pipeline_sheetsage2.py",
                    "processing_sheetsage2.py", "rendering_sheetsage2.py", "schema_sheetsage2.py",
                    "tensors_sheetsage2.py", "tokenization_sheetsage2.py", "config.json", "processor_config.json",
                    "model.safetensors", "requirements.txt", "LICENSE", "THIRD_PARTY_NOTICES.md"], 0.3),
    "MERT-v2-FullSong": ("m-a-p/MERT-v2-FullSong", "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
                         ["config.json", "configuration_mert2.py", "modeling_mert2.py", "model.safetensors",
                          "preprocessor_config.json", "weights_manifest.json", "LICENSE", "THIRD_PARTY_NOTICES.md"], 2.5),
    # Sung-lyrics ASR for WER/PER checks; the model WildSongBench's PER evaluator uses. Hub main on 2026-09-25.
    "Qwen3-ASR-1.7B": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5",
                       ["config.json", "generation_config.json", "chat_template.json", "merges.txt", "vocab.json",
                        "tokenizer_config.json", "preprocessor_config.json", "model.safetensors.index.json",
                        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "README.md"], 4.7),
}
GENERATION = ("YuE2-3B", "YuE2-Vae")
TRANSCRIPTION = ("SheetSage2", "MERT-v2-FullSong")
LYRICS = ("Qwen3-ASR-1.7B",)

TRANSCRIBE_WORKER = HERE / "workers" / "transcribe.py"
LYRICS_WORKER = HERE / "workers" / "lyrics.py"


def model_names(transcribe=True, legacy=False, lyrics=True):
    names = list(GENERATION)
    if legacy:
        names.append("YuE2-Vae-legacy")
    if transcribe:
        names += TRANSCRIPTION
    if lyrics:
        names += LYRICS
    return names


def hub_cache():
    """The Hugging Face hub cache directory, resolved like huggingface_hub does, without importing it."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).expanduser() / "hub"


def cache_snapshot(name):
    """<cache>/models--org--repo/snapshots/<revision> when every pinned file is there, else None."""
    repo, revision, files, _ = MODELS[name]
    if revision == "main":
        ref = hub_cache() / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        if not ref.is_file():
            return None
        revision = ref.read_text().strip()
    folder = hub_cache() / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if all((folder / f).is_file() for f in files):
        return folder
    return None


def import_local(source, name):
    """Link a `hf download --local-dir` snapshot at <source>/<name> into the cache, as hf would lay it out.

    Every file must carry the local-dir metadata (.cache/huggingface/download/<file>.metadata: commit,
    etag, time) naming the pinned commit; blobs are hard-linked (or copied across filesystems) under
    their etag, so a later `hf download` recognises them instead of fetching again.
    """
    import shutil
    repo, revision, files, _ = MODELS[name]
    src = Path(source).expanduser() / name
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is not a directory")
    entry = hub_cache() / f"models--{repo.replace('/', '--')}"
    commit = None
    plan = []
    for rel in files:
        file = src / rel
        if not file.is_file():
            raise FileNotFoundError(f"{name}: {rel} missing in {src}")
        meta = src / ".cache/huggingface/download" / (rel + ".metadata")
        if not meta.is_file():
            raise ValueError(f"{name}: {rel} has no download metadata; download it with `hf download --local-dir` first")
        lines = meta.read_text().splitlines()
        file_commit, etag = lines[0].strip(), lines[1].strip()
        if revision != "main" and file_commit != revision:
            raise ValueError(f"{name}: {rel} is from commit {file_commit[:12]}, pin is {revision[:12]}")
        if commit not in (None, file_commit):
            raise ValueError(f"{name}: files come from different commits ({commit[:12]} and {file_commit[:12]})")
        commit = file_commit
        plan.append((rel, file, etag))
    snapshot = entry / "snapshots" / commit
    for rel, file, etag in plan:
        blob = entry / "blobs" / etag
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            try:
                os.link(file, blob)
            except OSError:
                shutil.copy2(file, blob)
        link = snapshot / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(blob, link.parent), link)
    if revision == "main":
        (entry / "refs").mkdir(exist_ok=True)
        (entry / "refs" / "main").write_text(commit)
    return snapshot


def resolve(name):
    """Snapshot directory for a pinned model. Raises FileNotFoundError if it is not available offline.

    YUE2_MODELS_DIR=<dir> takes <dir>/<name> when that exists (the layout ../YuE/models and the YuE
    skills use); otherwise the Hugging Face cache at the pinned revision.
    """
    repo, revision, files, _ = MODELS[name]
    local = os.environ.get("YUE2_MODELS_DIR")
    if local:
        candidate = Path(local).expanduser() / name
        if candidate.is_dir():
            return candidate.resolve()
    folder = cache_snapshot(name)
    if folder is not None:
        return folder
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, revision=revision, local_files_only=True, allow_patterns=files))
    except Exception as exc:  # LocalEntryNotFoundError and friends
        raise FileNotFoundError(f"{name} ({repo}@{revision[:12]}) is not in the Hugging Face cache; "
                                f"run ./setup.sh (or set YUE2_MODELS_DIR to a directory holding {name}/)") from exc


def probe_gpu():
    """(name, total GiB) from nvidia-smi, or (None, 0) without a usable driver."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        name, mib = out.strip().splitlines()[0].rsplit(",", 1)
        return name.strip(), int(mib) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, 0.0


def choose_budget(total_gib):
    """memory_budget_gib for YuE2Pipeline. The pipeline reserves 2 GiB and caps at total-2 itself;
    28 keeps a 32 GB card from claiming everything, 24 is the documented baseline for a 24 GB card."""
    if total_gib <= 0:
        return 24
    return max(8, min(28, round(total_gib)))


def read_profile():
    if PROFILE.is_file():
        return json.loads(PROFILE.read_text())
    return None


def write_profile(data):
    PROFILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return PROFILE


def offline_env(base=None):
    env = dict(os.environ if base is None else base)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    return env


def _cli(argv):
    if not argv or argv[0] not in {"files", "names", "cached", "import"}:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "cached":
        folder = cache_snapshot(argv[1])
        print(folder or "")
        return 0 if folder else 1
    if argv[0] == "import":
        names = argv[2:] or model_names(transcribe=True, legacy=False, lyrics=False)
        failed = 0
        for name in names:
            try:
                print(f"{name}: -> {import_local(argv[1], name)}")
            except (OSError, ValueError, KeyError) as exc:
                print(f"{name}: skipped ({exc})", file=sys.stderr)
                failed += 1
        return 1 if failed else 0
    if argv[0] == "files":
        repo, revision, files, _ = MODELS[argv[1]]
        print(repo, revision, *files)
        return 0
    names = model_names(transcribe="--transcribe" in argv, legacy="--legacy" in argv, lyrics="--lyrics" in argv)
    print(" ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
