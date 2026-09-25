#!/usr/bin/env bash
# Download the pinned checkpoints into the Hugging Face cache, build the environments, and dry-run.
# The server never downloads anything (local_files_only), so run this first.
#
#   ./setup.sh                  # YuE2-3B + YuE2-Vae + SheetSage2 + MERT-v2-FullSong, envs, doctor, dry run
#   ./setup.sh --legacy         # also YuE2-Vae-legacy (benchmark-protocol decoder)
#   ./setup.sh --no-transcribe  # skip SheetSage2/MERT and their worker environment
#   ./setup.sh --no-lyrics      # skip Qwen3-ASR (sung-lyrics recognition, WER/PER) and its worker environment
#   ./setup.sh --dry-run        # show what would be downloaded
#   ./setup.sh --no-sync        # skip building the uv environments
#   ./setup.sh --no-verify      # skip `yue2 doctor` and the dry run (no profile is written)
#   ./setup.sh --load-only      # dry run loads the models but generates nothing
#   ./setup.sh --import <dir>   # first link <dir>/<name>/ snapshots (hf --local-dir layout, e.g. a YuE
#                               # checkout's models/) into the cache instead of downloading them
#
# Files already in the cache are skipped. HF_HOME / HF_HUB_CACHE / HF_TOKEN are respected.
# Revisions are pinned in yue2_common.py; the server resolves exactly those snapshots.
set -euo pipefail

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

legacy=0 transcribe=1 lyrics=1 sync=1 verify=1 load_only=0
dry_run=()
import_dir=""
while (($#)); do
    case "$1" in
        --legacy) legacy=1; shift ;;
        --no-transcribe) transcribe=0; shift ;;
        --no-lyrics) lyrics=0; shift ;;
        --dry-run) dry_run=(--dry-run); shift ;;
        --no-sync) sync=0; shift ;;
        --no-verify) verify=0; shift ;;
        --load-only) load_only=1; shift ;;
        --import) [[ $# -ge 2 ]] || usage 1; import_dir=$2; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

cd "$(dirname "$0")"

for tool in uv hf python3; do
    command -v "$tool" >/dev/null || {
        echo "$tool not found. Install uv from https://docs.astral.sh/uv/, then: uv tool install huggingface_hub" >&2
        exit 1
    }
done
if ((transcribe)) && ! command -v ffmpeg >/dev/null; then
    echo "ffmpeg not found; SheetSage2 needs it (apt install ffmpeg). Use --no-transcribe for generation only." >&2
    exit 1
fi
if ! nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >/dev/null 2>&1; then
    echo "warning: nvidia-smi not usable; downloads proceed, but the dry run needs a CUDA GPU" >&2
fi

# ── Checkpoints ────────────────────────────────────────────────────────────────
names=()
((legacy)) && names+=(--legacy)
((transcribe)) && names+=(--transcribe)
((lyrics)) && names+=(--lyrics)
read -r -a models <<<"$(python3 yue2_common.py names "${names[@]}")"

if [[ -n $import_dir ]]; then
    echo "== importing local snapshots from $import_dir into the Hugging Face cache (hard links)"
    python3 yue2_common.py import "$import_dir" "${models[@]}" || echo "   (models not imported are downloaded below)"
fi
for name in "${models[@]}"; do
    read -r repo revision files <<<"$(python3 yue2_common.py files "$name")"
    read -r -a file_list <<<"$files"
    if cached=$(python3 yue2_common.py cached "$name"); then
        echo "== $name: already in the cache ($cached)"
        continue
    fi
    echo "== $name: $repo @ ${revision:0:12} (${#file_list[@]} files)"
    # Explicit filenames: hf download fetches exactly these into the cache snapshot for that revision.
    hf download "$repo" "${file_list[@]}" --revision "$revision" "${dry_run[@]}" >/dev/null
done
((${#dry_run[@]})) && { echo "dry run: nothing downloaded"; exit 0; }

# ── Environments ───────────────────────────────────────────────────────────────
if ((sync)); then
    echo "== uv sync (server: yue2-infer, torch 2.10, fastapi)"
    uv sync
    if ((transcribe)); then
        echo "== uv sync --script workers/transcribe.py (SheetSage2: torch 2.8 cu128, transformers 4.45)"
        uv sync --script workers/transcribe.py --locked
    fi
    if ((lyrics)); then
        echo "== uv sync --script workers/lyrics.py (Qwen3-ASR: torch 2.10, transformers 4.57.6, g2p_en, pypinyin)"
        uv sync --script workers/lyrics.py --locked
        echo "== nltk data for g2p_en (offline at run time)"
        uv run --locked --script workers/lyrics.py --prepare
    fi
fi

((verify)) || { echo "skipped doctor and dry run (--no-verify); no profile written"; exit 0; }

# ── Verify ─────────────────────────────────────────────────────────────────────
# Environment report only: doctor's --verify-hashes resolves the models without a revision, which needs a
# `main` ref the commit-pinned cache does not have. The dry run below verifies every weight hash instead.
echo "== yue2 doctor (dependency versions, CUDA devices)"
HF_HUB_OFFLINE=1 uv run yue2 doctor --offline >/dev/null
echo "   ok"

echo "== dry run (yue2_dryrun.py --write-profile)"
flags=(--write-profile)
((load_only)) && flags+=(--no-generate)
((transcribe)) || flags+=(--no-transcribe)
HF_HUB_OFFLINE=1 uv run yue2_dryrun.py "${flags[@]}"

echo
echo "Done. Profile: yue2_profile.json. Start the server with:"
echo "  uv run yue2_server.py"
echo "Rerun the check any time with: uv run yue2_dryrun.py"
