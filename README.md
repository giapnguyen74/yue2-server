# yue2-server

HTTP job server for [YuE2](https://github.com/multimodal-art-projection/YuE) song generation on one GPU.
It loads `YuE2Pipeline` once, queues JSON requests, runs them one at a time, and serves the native
artifacts by job id. SheetSage2 transcription (audio → ABC) and Qwen3-ASR lyrics recognition (audio →
transcript, WER/CER and PER against the intended lyrics) are served through the same queue, so a client
needs no GPU, no torch and no model files. The `skills/yue2-music-server/` agent skill is the intended
client; `scripts/yue2-client.sh` is the same contract in curl.

Sized for a 24 GB BF16-capable NVIDIA card (the YuE2 baseline); a 32 GB card gets a larger budget.

## Setup

Requires [uv](https://docs.astral.sh/uv/), the `hf` CLI (`uv tool install huggingface_hub`), `ffmpeg`
for transcription, and an NVIDIA driver for CUDA 12.8.

```bash
./setup.sh                 # pinned snapshots into the HF cache, both uv envs, doctor, dry run -> yue2_profile.json
./setup.sh --legacy        # also YuE2-Vae-legacy, the benchmark-protocol decoder
./setup.sh --no-transcribe # skip SheetSage2
./setup.sh --no-lyrics     # skip Qwen3-ASR
./setup.sh --dry-run       # show what would be downloaded
```

The pins live in `yue2_common.py` (the commits the YuE uv skill records). Snapshots you already have from
`hf download --local-dir`, laid out as `<dir>/YuE2-3B`, `<dir>/YuE2-Vae`, ... (for example a YuE checkout's
`models/`), go into the cache without a second download:

```bash
./setup.sh --import /path/to/YuE/models      # hard-links the files into the HF cache, then continues as usual
```

The importer reads each file's `--local-dir` metadata, refuses files from a commit other than the pin, and
lays the cache out exactly as `hf` would (blobs by etag, symlinked snapshots), so `hf cache ls` shows them
and a later `hf download` re-uses them. Alternatively point the server straight at such a directory with
`YUE2_MODELS_DIR=<dir>`.

The dry run loads and hash-verifies both checkpoints, generates a 20 s song, transcribes it with the
SheetSage2 worker, and records the GPU, budget and measured rates in `yue2_profile.json`. Those rates are
the server's first ETA priors. Rerun it any time with `uv run yue2_dryrun.py`.

## Run

```bash
uv run yue2_server.py                     # 127.0.0.1:8001, budget and decoder from the profile
uv run yue2_server.py --port 8001 --budget 28 --vae standard --keep-jobs 50 --no-warmup
```

Every job's directory under `outputs/<id>/` is exactly what YuE2's `save_artifacts()` (or `plan.save()`,
or the SheetSage2 worker) wrote, so `yue2.storage.verify_result` and the YuE skill's tools accept it. The
one addition is `audio.mp3`, encoded with ffmpeg from the lossless `audio.flac` once the manifests are
final; the FLAC stays the hash-verified artifact and the MP3 is the delivery copy.
Finished jobs are re-registered after a restart; `--keep-jobs N` prunes the oldest beyond N.

## API

| Method, path | |
|---|---|
| `POST /jobs` | JSON `{task: generate \| plan \| decode, ...}`; `202` with `id`, `position`, `eta_s`, `seed` |
| `POST /jobs/transcribe` | multipart `audio=@file` plus `task` (`full`, `melody-full`, `melody-vocal`), `preset`, `max_seconds`, `dtype` |
| `POST /jobs/lyrics` | multipart `audio=@file` plus `lyrics` (reference text, optional), `language` (`auto`, `English`, `Chinese`), `passes` (1–8); or JSON `{task: lyrics, source_job, lyrics, language, passes}` to score a finished job's audio |
| `GET /jobs/{id}` | `queued` / `running` / `done` / `failed` / `cancelled`, with `phase`, `tokens`, `eta_s`, and `artifacts` when done |
| `GET /jobs/{id}/wait?timeout=300` | long-poll: returns when the job is terminal or after `timeout` seconds |
| `GET /jobs/{id}/audio` | `audio.mp3` by default; `?format=flac` for the lossless artifact, `?format=wav`; `409` until done, `404` for plan/transcribe jobs |
| `GET /jobs/{id}/score` | `score.abc` as text |
| `GET /jobs/{id}/artifacts/{name}` | any file the status view lists under `artifacts`, unchanged |
| `DELETE /jobs/{id}` | cancel a queued job, or stop a running one at its next token / ODE step |
| `GET /jobs`, `GET /health` | all jobs newest first; `running` and `queued` for a router's busy check |

`lyrics` jobs run Qwen3-ASR-1.7B, the model WildSongBench's PER evaluator uses, and write `transcript.txt`
and `lyrics_asr.json` with every pass, the detected language, the unit error rate (WER for English, CER for
Chinese) and the phoneme error rate against the reference. The scoring (lower-cased, section tags and
punctuation dropped; ARPAbet via g2p_en for Latin words, tone-less pinyin initial+final for CJK) is
documented in the report; it is not the benchmark's undisclosed evaluator, so compare versions with each
other.

`generate` and `plan` take the YuE2 request fields: `style` (alias `tags`), `lyrics`, `cot`
(`full` \| `melody` \| `off`), `seed` (random when omitted), `abc` (a supplied score, needs `full` or
`melody`), `cfg_scale`, `id`, and optional `abc_sampling` / `semantic_sampling` overrides. `plan` returns
only the score. `decode` takes `source_job` and `vae` (`standard` \| `legacy`) and re-decodes a finished
job's cached latents.

```bash
curl -s -X POST localhost:8001/jobs -H 'content-type: application/json' -d '{
  "style": "English, warm piano pop, expressive female voice, 88 BPM",
  "lyrics": "[Verse]\nNeon fades along the lane\n[Chorus]\nLet the day come into view\n",
  "cot": "full", "seed": 831001}'                      # -> {"id": "e3969fd18827", ...}
curl -s 'localhost:8001/jobs/e3969fd18827/wait?timeout=600' | jq .status
curl -s -o song.mp3 localhost:8001/jobs/e3969fd18827/audio            # ?format=flac for lossless
curl -s localhost:8001/jobs/e3969fd18827/score
curl -s -X POST localhost:8001/jobs/transcribe -F audio=@song.mp3 -F task=melody-full
```

Or `scripts/yue2-client.sh song request.json outdir`.

Phases of a `generate` job: `planning` (ABC tokens), `semantic` (codec tokens), `synthesizing` (NAR flow
matching), `decoding` (VAE), `saving`. Cancellation lands within a token or ODE step; there is no hook
inside the VAE decode, so a cancel there takes effect right after it. Transcription runs
`workers/transcribe.py` as a subprocess in its own uv environment (torch 2.8, transformers 4.45) after
moving the YuE2 model off the GPU; phases are `loading` and `transcribing`. Lyrics recognition runs
`workers/lyrics.py` the same way (qwen-asr pins accelerate 1.12 against yue2-infer's 1.13); phases are
`loading` and `recognizing`.

## Router

For [simple-ai-router](../simple-ai-router):

```yaml
yue2:
  cmd: uv run --directory /path/to/yue2-server yue2_server.py --port ${PORT}
  busyCheck: {endpoint: /health, fields: [running, queued]}
  drainTimeout: 30m
```

## Tests

```bash
uv sync --extra test && uv run pytest
```

The suite runs a fake pipeline (real YuE2 artifact classes, no GPU) and a fake transcription worker, and
drives the skill's stdlib clients against a live uvicorn. `uv run yue2_dryrun.py` is the real-model check.

## Files

| | |
|---|---|
| `yue2_server.py` | FastAPI app and entry point |
| `yue2_jobs.py` | job queue, worker thread, the four tasks, ETA, recovery, retention |
| `yue2_schemas.py` | request body model |
| `yue2_common.py` | model pins, offline resolve, GPU probe, profile |
| `yue2_dryrun.py`, `setup.sh` | setup and real-model check |
| `workers/` | SheetSage2 worker (from the YuE uv skill) and the Qwen3-ASR lyrics worker, each with its own lockfile |
| `skills/yue2-music-server/` | agent skill whose helpers call this server |
| `scripts/yue2-client.sh` | curl/jq client |
| `PLAN.md` | design notes and the HTTP contract |

## License

Server code: Apache 2.0, like the YuE2 runtime and skill it wraps. YuE2, SheetSage2 and MERT weights are
CC BY-NC 4.0 and keep their own terms; Qwen3-ASR is Apache 2.0.
