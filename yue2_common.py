"""Shared by setup.sh, yue2_dryrun.py and the server: pinned model snapshots, the profile, GPU sizing.

The pins are the commits the YuE repository's uv skill records. Every model lives in the Hugging Face
cache (HF_HOME / HF_HUB_CACHE respected) and is resolved offline; nothing here downloads.

  python3 yue2_common.py files YuE2-3B        # "<repo> <revision> <file>..." for `hf download` (stdlib only)
  python3 yue2_common.py names [--transcribe] [--legacy]
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
    # Benchmark-protocol decoder; optional. No reviewed pin recorded in the YuE skills, so main.
    "YuE2-Vae-legacy": ("m-a-p/YuE2-Vae-legacy", "main", _VAE_FILES, 0.5),
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
}
GENERATION = ("YuE2-3B", "YuE2-Vae")
TRANSCRIPTION = ("SheetSage2", "MERT-v2-FullSong")

TRANSCRIBE_WORKER = HERE / "workers" / "transcribe.py"


def model_names(transcribe=True, legacy=False):
    names = list(GENERATION)
    if legacy:
        names.append("YuE2-Vae-legacy")
    if transcribe:
        names += TRANSCRIPTION
    return names


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
    if not argv or argv[0] not in {"files", "names"}:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "files":
        repo, revision, files, _ = MODELS[argv[1]]
        print(repo, revision, *files)
        return 0
    names = model_names(transcribe="--transcribe" in argv, legacy="--legacy" in argv)
    print(" ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
