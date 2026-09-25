"""yue2-server: HTTP job queue in front of one YuE2Pipeline (and the SheetSage2 worker).

  uv run yue2_server.py                       # loads the profile from ./setup.sh, listens on 127.0.0.1:8001
  uv run yue2_server.py --port 8001 --budget 28 --vae standard --no-warmup

  POST   /jobs                 JSON {task: generate|plan|decode, ...}; 202 {id, status, position, eta_s}
  POST   /jobs/transcribe      multipart: audio=@file, task=full|melody-full|melody-vocal, preset, max_seconds, dtype
  POST   /jobs/lyrics          multipart: audio=@file, lyrics (reference text), language, passes; or JSON task=lyrics with source_job
  GET    /jobs/{id}            status view; GET /jobs/{id}/wait?timeout=300 long-polls until terminal
  GET    /jobs/{id}/audio      audio.mp3 by default (?format=flac for the lossless artifact, ?format=wav)
  GET    /jobs/{id}/artifacts/{name}   any file the status view lists under "artifacts"
  DELETE /jobs/{id}            cancel
  GET    /jobs, GET /health    all jobs newest first; running/queued for simple-ai-router's busyCheck
"""

from __future__ import annotations

import argparse
import asyncio
import io
import random
import shutil
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response

from yue2_common import GENERATION, LYRICS, MODELS, choose_budget, probe_gpu, read_profile, resolve
from yue2_jobs import AUDIO_SUFFIXES, JobQueue, audio_duration, convert_mp3
from yue2_schemas import SubmitRequest

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_pipeline(budget, vae="standard", warmup=True, progress=False):
    from yue2 import YuE2Pipeline
    paths = {name: resolve(name) for name in GENERATION}
    vae_path = resolve("YuE2-Vae-legacy") if vae == "legacy" else paths["YuE2-Vae"]
    t0 = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained(str(paths["YuE2-3B"]), vae=str(vae_path), device="cuda",
                                        memory_budget_gib=budget, local_files_only=True, progress=progress)
    print(f"pipeline ready in {time.perf_counter() - t0:.1f}s: {paths['YuE2-3B']}", flush=True)
    if warmup:
        loader = getattr(pipe, "_load_model", None)
        if loader is not None:
            t0 = time.perf_counter()
            loader()
            print(f"MoT model on GPU in {time.perf_counter() - t0:.1f}s", flush=True)
    return pipe


def create_app(jq, settings):
    app = FastAPI(title="yue2-server", version="0.1.0")
    max_upload = settings.get("max_upload_mb", 200) * 1024 * 1024

    def get(job_id):
        job = jq.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    def accepted(job):
        view = jq.view(job)
        return {"id": job.id, "task": job.task, "status": job.status, "position": view["position"],
                "eta_s": view["eta_s"], "seed": view["seed"]}

    def lyrics_request(lyrics, language, passes):
        if language.lower() not in ("auto", "english", "chinese"):
            raise HTTPException(422, "language must be auto, English or Chinese")
        if not 1 <= passes <= 8:
            raise HTTPException(422, "passes must be 1-8")
        if lyrics is not None and not lyrics.strip():
            lyrics = None
        return {"lyrics": lyrics, "language": language, "passes": passes}

    async def save_upload(job, audio):
        name = Path(audio.filename or "audio").name
        if Path(name).suffix.lower() not in AUDIO_SUFFIXES:
            shutil.rmtree(job.dir, ignore_errors=True)
            raise HTTPException(415, f"unsupported audio type; use one of {sorted(AUDIO_SUFFIXES)}")
        target = job.dir / "upload" / name
        target.parent.mkdir(parents=True)
        size = 0
        try:
            with target.open("wb") as stream:
                while chunk := await audio.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload:
                        raise HTTPException(413, f"audio exceeds {settings.get('max_upload_mb', 200)} MB")
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(422, "empty upload")
        except HTTPException:
            shutil.rmtree(job.dir, ignore_errors=True)
            raise
        job.request["filename"] = name
        job.request["bytes"] = size
        duration = audio_duration(target)
        if duration:
            job.request["audio_seconds"] = round(duration, 1)
        return job

    @app.post("/jobs/lyrics", status_code=202)
    async def lyrics(audio: UploadFile = File(...), lyrics: str | None = Form(None), language: str = Form("auto"),
                     passes: int = Form(1)):
        if not jq.lyrics_enabled:
            raise HTTPException(422, "lyrics recognition is not enabled on this server")
        job = jq.new_job("lyrics", lyrics_request(lyrics, language, passes))
        return accepted(jq.submit(await save_upload(job, audio)))

    @app.post("/jobs", status_code=202)
    def submit(req: SubmitRequest):
        from yue2.protocol import SongRequest, resolve_sampling, GenerationConfig
        body = req.model_dump()
        task = body.pop("task")
        source_job, vae = body.pop("source_job"), body.pop("vae")
        language, passes = body.pop("language"), body.pop("passes")
        if task == "lyrics":
            if not jq.lyrics_enabled:
                raise HTTPException(422, "lyrics recognition is not enabled on this server")
            if not source_job:
                raise HTTPException(422, "lyrics as JSON needs source_job; upload audio to /jobs/lyrics instead")
            source = jq.get(source_job)
            if source is None or source.status != "done" or not (source.artifact_root / "audio.flac").is_file():
                raise HTTPException(422, "source_job must be a finished job with audio")
            request = lyrics_request(body.get("lyrics"), language, passes)
            request["source_job"] = source_job
            if source.result and source.result.get("audio_seconds"):
                request["audio_seconds"] = source.result["audio_seconds"]
            return accepted(jq.submit(jq.new_job("lyrics", request)))
        if task == "decode":
            if not source_job:
                raise HTTPException(422, "decode needs source_job")
            source = jq.get(source_job)
            if source is None or source.status != "done" or source.task not in ("generate", "decode"):
                raise HTTPException(422, "source_job must be a finished generate or decode job")
            if vae == "legacy" and not jq.legacy_vae:
                raise HTTPException(422, "the legacy decoder is not installed; run ./setup.sh --legacy")
            import numpy as np
            frames = int(np.load(source.artifact_root / "semantic.npy", allow_pickle=False, mmap_mode="r").shape[0])
            request = {"source_job": source_job, "vae": vae, "frames": frames}
            return accepted(jq.submit(jq.new_job("decode", request)))

        style = body.pop("tags") if body.get("style") is None else body.pop("style")
        if body.get("tags") is not None and body["tags"] != style:
            raise HTTPException(422, "style and tags are aliases and cannot disagree")
        body.pop("tags", None)
        if style is None or body.get("lyrics") is None:
            raise HTTPException(422, "style and lyrics are required")
        if body["seed"] is None:
            body["seed"] = random.randrange(2**31)
        if task == "plan" and body["cot"] == "off":
            raise HTTPException(422, "plan needs cot=full or cot=melody; cot=off has no score")
        request = {"style": style, "lyrics": body["lyrics"], "cot": body["cot"], "seed": body["seed"],
                   "abc": body["abc"], "cfg_scale": body["cfg_scale"], "id": body["id"]}
        try:
            SongRequest(**request)
            defaults = GenerationConfig()
            for key in ("abc_sampling", "semantic_sampling"):
                if body[key] is not None:
                    resolve_sampling(body[key], getattr(defaults, "abc" if key == "abc_sampling" else "semantic"))
                    request[key] = body[key]
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None
        return accepted(jq.submit(jq.new_job(task, request)))

    @app.post("/jobs/transcribe", status_code=202)
    async def transcribe(audio: UploadFile = File(...), task: str = Form("full"), preset: str = Form("default"),
                         max_seconds: float | None = Form(None), dtype: str = Form("bf16")):
        if not jq.transcribe_enabled:
            raise HTTPException(422, "transcription is not enabled on this server (setup ran with --no-transcribe)")
        if task not in ("full", "melody-full", "melody-vocal"):
            raise HTTPException(422, "task must be full, melody-full or melody-vocal")
        if preset not in ("default", "paper") or dtype not in ("bf16", "fp32"):
            raise HTTPException(422, "preset must be default|paper and dtype bf16|fp32")
        if max_seconds is not None and max_seconds <= 0:
            raise HTTPException(422, "max_seconds must be positive")
        job = jq.new_job("transcribe", {"task": task, "preset": preset, "max_seconds": max_seconds, "dtype": dtype})
        return accepted(jq.submit(await save_upload(job, audio)))

    @app.get("/jobs")
    def list_jobs():
        with jq.lock:
            jobs = [jq.jobs[j] for j in reversed(jq.order)]
        return [jq.summary(j) for j in jobs]

    @app.get("/jobs/{job_id}")
    def status(job_id: str):
        return jq.view(get(job_id))

    @app.get("/jobs/{job_id}/wait")
    async def wait(job_id: str, timeout: float = Query(300, ge=0, le=3600)):
        job = get(job_id)
        if not job.terminal:
            await asyncio.get_running_loop().run_in_executor(None, job.done_event.wait, timeout)
        return jq.view(job)

    @app.get("/jobs/{job_id}/audio")
    def audio_file(job_id: str, format: str = Query("mp3")):
        job = get(job_id)
        if job.status != "done":
            raise HTTPException(409, f"job is {job.status}")
        path = job.artifact_root / "audio.flac"
        if not path.is_file():
            raise HTTPException(404, "this job has no audio")
        if format == "mp3":
            try:
                mp3 = convert_mp3(path)
            except (subprocess.CalledProcessError, OSError) as exc:
                raise HTTPException(500, f"mp3 conversion failed: {exc}") from None
            if mp3 is None:
                raise HTTPException(501, "mp3 needs ffmpeg on the server; use ?format=flac or wav")
            return FileResponse(mp3, media_type="audio/mpeg", filename=f"{job.id}.mp3")
        if format == "flac":
            return FileResponse(path, media_type="audio/flac", filename=f"{job.id}.flac")
        if format != "wav":
            raise HTTPException(422, "format must be mp3, flac or wav")
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        buffer = io.BytesIO()
        sf.write(buffer, data, rate, format="WAV", subtype="PCM_24")
        return Response(buffer.getvalue(), media_type="audio/wav",
                        headers={"content-disposition": f'attachment; filename="{job.id}.wav"'})

    @app.get("/jobs/{job_id}/score")
    def score(job_id: str):
        job = get(job_id)
        if job.status != "done":
            raise HTTPException(409, f"job is {job.status}")
        path = job.artifact_root / "score.abc"
        if not path.is_file():
            raise HTTPException(404, "this job has no score")
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @app.get("/jobs/{job_id}/artifacts/{name:path}")
    def artifact(job_id: str, name: str):
        job = get(job_id)
        if job.status != "done":
            raise HTTPException(409, f"job is {job.status}")
        if not job.artifacts or name not in job.artifacts:
            raise HTTPException(404, "not an artifact of this job")
        path = job.artifact_root / name
        if not path.is_file():
            raise HTTPException(404, "artifact missing on disk")
        media = {"json": "application/json", "abc": "text/plain; charset=utf-8", "flac": "audio/flac",
                 "npy": "application/octet-stream", "mid": "audio/midi", "lab": "text/plain; charset=utf-8",
                 "txt": "text/plain; charset=utf-8"}.get(path.suffix.lstrip("."), "application/octet-stream")
        return FileResponse(path, media_type=media, filename=path.name)

    @app.delete("/jobs/{job_id}")
    def cancel(job_id: str):
        job = get(job_id)
        jq.cancel(job)
        return {"id": job.id, "status": job.status, "cancel_pending": job.cancel.is_set() and not job.terminal}

    @app.get("/health")
    def health():
        data = {"status": "ok", **jq.health(), **settings["static"]}
        return data

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--vae", choices=("standard", "legacy"), default="standard",
                    help="decoder for generate jobs; decode jobs choose per request")
    ap.add_argument("--budget", type=float, help="memory_budget_gib; default from the profile, else the GPU")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--keep-jobs", type=int, default=0, help="delete the oldest finished job directories beyond N (0 = keep all)")
    ap.add_argument("--max-upload-mb", type=int, default=200)
    ap.add_argument("--no-warmup", action="store_true", help="do not move the 3B model to the GPU before the first job")
    ap.add_argument("--no-transcribe", action="store_true", help="refuse transcription jobs")
    ap.add_argument("--no-lyrics", action="store_true", help="refuse lyrics recognition (Qwen3-ASR) jobs")
    ap.add_argument("--progress", action="store_true", help="show YuE2's own stderr progress lines")
    args = ap.parse_args()

    profile = read_profile() or {}
    gpu_name, total = probe_gpu()
    budget = args.budget or profile.get("budget_gib") or choose_budget(total)
    print(f"GPU: {gpu_name or 'none detected'} ({total:.1f} GiB); budget {budget} GiB; "
          f"profile: {'yes' if profile else 'none (run ./setup.sh)'}", flush=True)

    pipe = build_pipeline(budget, vae=args.vae, warmup=not args.no_warmup, progress=args.progress)
    try:
        legacy = resolve("YuE2-Vae-legacy")
    except FileNotFoundError:
        legacy = None
    transcribe = not args.no_transcribe
    if transcribe:
        try:
            for name in ("SheetSage2", "MERT-v2-FullSong"):
                resolve(name)
        except FileNotFoundError as exc:
            print(f"transcription disabled: {exc}", flush=True)
            transcribe = False
        if transcribe and not shutil.which("uv"):
            print("transcription disabled: uv not on PATH", flush=True)
            transcribe = False

    lyrics = not args.no_lyrics
    if lyrics:
        try:
            resolve(LYRICS[0])
        except FileNotFoundError as exc:
            print(f"lyrics recognition disabled: {exc}", flush=True)
            lyrics = False
        if lyrics and not shutil.which("uv"):
            lyrics = False
    jq = JobQueue(pipe, out_dir=args.output_dir, gpu=gpu_name, profile=profile, vae=args.vae,
                  keep_jobs=args.keep_jobs, transcribe=transcribe, lyrics=lyrics, legacy_vae=legacy)
    static = {
        "gpu": gpu_name, "vram_gib": round(total, 1), "budget_gib": budget, "vae": args.vae,
        "models": {name: {"repo": MODELS[name][0], "revision": MODELS[name][1]} for name in GENERATION},
        "weights": {"mot": pipe.weights["mot"]["config_sha256"], "vae": pipe.weights["vae"]["config_sha256"]},
        "output_dir": str(args.output_dir), "max_upload_mb": args.max_upload_mb,
    }
    app = create_app(jq, {"static": static, "max_upload_mb": args.max_upload_mb})
    print(f"yue2-server listening on http://{args.host}:{args.port} (recovered {len(jq.jobs)} finished jobs)", flush=True)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
