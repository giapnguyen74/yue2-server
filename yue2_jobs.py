"""Job queue for yue2-server: one worker thread owns the GPU and runs jobs serially.

Tasks:
  generate    style + lyrics [+ abc] -> song; the four staged pipeline calls, mirroring YuE2Pipeline.__call__
  plan        style + lyrics -> ABC score only
  decode      a finished job's cached latents -> audio again, with the standard or legacy decoder
  transcribe  uploaded audio -> ABC via the SheetSage2 worker (workers/transcribe.py) in its own uv env

Every job directory is exactly what YuE2's own save_artifacts / plan.save / the worker wrote, plus a
server_job.json written after the manifests so it never enters them.
"""

from __future__ import annotations

import json
import queue
import secrets
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from yue2_common import HERE, LYRICS, LYRICS_WORKER, MODELS, TRANSCRIBE_WORKER, TRANSCRIPTION, offline_env, resolve

TERMINAL = {"done", "failed", "cancelled"}
TIMING_FILE = HERE / "yue2_timing.json"
# Before any measurement. The dry run (yue2_profile.json) replaces most of these on first start.
PRIORS = {"abc_tok_s": 30.0, "semantic_tok_s": 30.0, "nar_s_per_token": 0.02, "vae_s_per_token": 0.004,
          "overhead_s": 15.0, "abc_len": 1500.0, "semantic_len": 6000.0,
          "transcribe_load_s": 45.0, "transcribe_s_per_audio_s": 0.2,
          "lyrics_load_s": 30.0, "lyrics_s_per_audio_s": 0.1}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac"}
MP3_QUALITY = "2"        # libmp3lame VBR level: ~190 kbit/s


class WorkerError(RuntimeError):
    """The transcription worker failed; the message is already the user-facing error."""


def _now():
    return time.time()


@dataclass
class Job:
    id: str
    task: str
    request: dict                       # validated client fields (seed filled in) for the status view
    dir: Path
    created: float = field(default_factory=_now)
    status: str = "queued"              # queued | running | done | failed | cancelled
    phase: str | None = None            # planning | semantic | synthesizing | decoding | saving | loading | transcribing
    phase_started: float | None = None
    tokens: int = 0                     # emitted in the current AR phase
    abc_tokens: int = 0
    semantic_tokens: int = 0
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    artifacts: list[str] | None = None  # downloadable file names, relative to artifact_root
    result: dict | None = None          # summary from result.json / plan.json / transcription manifest
    recovered: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    proc: subprocess.Popen | None = None

    @property
    def artifact_root(self):
        if self.task == "transcribe":
            return self.dir / "transcription"
        if self.task == "lyrics":
            return self.dir / "lyrics"
        return self.dir

    @property
    def terminal(self):
        return self.status in TERMINAL


class Timing:
    """Measured per-phase rates, EMA-smoothed and persisted per GPU so restarts keep them."""

    def __init__(self, gpu, path=TIMING_FILE, profile=None):
        self.gpu = gpu or "unknown"
        self.path = Path(path)
        self.values = dict(PRIORS)
        self.source = "prior"
        self.measured = set()           # keys that hold a real measurement (not a prior or dry-run seed)
        self._lock = threading.Lock()
        saved = {}
        if self.path.is_file():
            try:
                saved = json.loads(self.path.read_text()).get(self.gpu, {})
            except (OSError, ValueError):
                saved = {}
        if saved:
            self.values.update(saved)
            self.measured = set(saved)
            self.source = "measured"
        elif profile:
            self._seed_from_profile(profile)

    def _seed_from_profile(self, profile):
        gen = profile.get("generate") or {}
        seeded = False
        for key in ("abc_tok_s", "semantic_tok_s", "nar_s_per_token", "vae_s_per_token"):
            if gen.get(key):
                self.values[key] = float(gen[key])
                seeded = True
        load = profile.get("load") or {}
        if load.get("mot_to_gpu_s"):
            self.values["overhead_s"] = float(load["mot_to_gpu_s"]) + 5.0
        tr = profile.get("transcribe") or {}
        if tr.get("model_loaded") and tr.get("elapsed_s"):
            self.values["transcribe_load_s"] = min(float(tr["elapsed_s"]), self.values["transcribe_load_s"])
        if seeded:
            self.source = "dryrun"

    def get(self, key):
        with self._lock:
            return self.values[key]

    def update(self, **measured):
        with self._lock:
            for key, value in measured.items():
                if value is None or not np.isfinite(value) or value <= 0:
                    continue
                # The first real measurement replaces a prior or dry-run seed; later ones are smoothed.
                if key in self.measured:
                    self.values[key] = 0.7 * self.values[key] + 0.3 * value
                else:
                    self.values[key] = value
                    self.measured.add(key)
            self.source = "measured"
            data = {}
            if self.path.is_file():
                try:
                    data = json.loads(self.path.read_text())
                except (OSError, ValueError):
                    data = {}
            data[self.gpu] = self.values
            try:
                self.path.write_text(json.dumps(data, indent=2) + "\n")
            except OSError:
                pass


class JobQueue:
    def __init__(self, pipe, *, out_dir, gpu=None, profile=None, vae="standard", keep_jobs=0,
                 timing_path=TIMING_FILE, worker=TRANSCRIBE_WORKER, worker_command=None, transcribe=True,
                 lyrics_worker=LYRICS_WORKER, lyrics=True, legacy_vae=None, start_worker=True):
        self.pipe = pipe
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vae = vae
        self.keep_jobs = keep_jobs
        self.worker_script = Path(worker)
        self.worker_command = list(worker_command or ["uv", "run", "--locked", "--script"])
        self.transcribe_enabled = transcribe
        self.lyrics_worker = Path(lyrics_worker)
        self.lyrics_enabled = lyrics
        self.legacy_vae = legacy_vae            # Path or None
        self._legacy_identity = None
        self.timing = Timing(gpu, timing_path, profile)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.queue: queue.Queue[str] = queue.Queue()
        self.running: str | None = None
        self.lock = threading.RLock()
        self._recover()
        if start_worker:
            threading.Thread(target=self._worker, name="yue2-worker", daemon=True).start()

    # ── submission and views ──────────────────────────────────────────────────

    def new_job(self, task, request):
        job_id = secrets.token_hex(6)
        job = Job(job_id, task, request, self.out_dir / job_id)
        job.dir.mkdir(parents=True, exist_ok=False)
        return job

    def submit(self, job):
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.queue.put(job.id)
        return job

    def get(self, job_id):
        return self.jobs.get(job_id)

    def position(self, job):
        with self.lock:
            if job.status != "queued":
                return 0
            waiting = [j for j in self.order if self.jobs[j].status == "queued"]
            return waiting.index(job.id) + 1 if job.id in waiting else 0

    def cancel(self, job):
        with self.lock:
            if job.terminal:
                return False
            if job.status == "queued":
                self._finish(job, "cancelled", None)
                return True
            job.cancel.set()
            proc = job.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        return True

    def wait(self, job, timeout):
        job.done_event.wait(timeout)
        return job

    def view(self, job):
        now = _now()
        elapsed = None
        if job.started is not None:
            elapsed = round((job.finished or now) - job.started, 1)
        data = {
            "id": job.id, "task": job.task, "status": job.status, "phase": job.phase,
            "tokens": job.tokens, "abc_tokens": job.abc_tokens, "semantic_tokens": job.semantic_tokens,
            "position": self.position(job), "eta_s": self.eta(job), "eta_source": self.timing.source,
            "created": job.created, "started": job.started, "finished": job.finished, "elapsed_s": elapsed,
            "request": job.request, "seed": job.request.get("seed"),
            "cancel_pending": job.cancel.is_set() and not job.terminal,
            "error": job.error, "artifacts": job.artifacts, "recovered": job.recovered,
            "truncated": None, "audio_seconds": None, "identity": None, "timing": None, "warnings": None,
            "audio_url": None, "score_url": None,
        }
        if job.status == "done" and job.result:
            for key in ("truncated", "audio_seconds", "identity", "timing", "warnings"):
                if key in job.result:
                    data[key] = job.result[key]
            if (job.artifact_root / "audio.flac").is_file():
                data["audio_url"] = f"/jobs/{job.id}/audio"
            if (job.artifact_root / "score.abc").is_file():
                data["score_url"] = f"/jobs/{job.id}/score"
        return data

    def summary(self, job):
        return {"id": job.id, "task": job.task, "status": job.status, "phase": job.phase,
                "created": job.created, "finished": job.finished, "eta_s": self.eta(job)}

    def health(self):
        with self.lock:
            queued = sum(1 for j in self.jobs.values() if j.status == "queued")
        return {"running": self.running, "queued": queued, "jobs": len(self.jobs),
                "timing": dict(self.timing.values), "eta_source": self.timing.source,
                "decoders": ["standard"] + (["legacy"] if self.legacy_vae else []),
                "transcribe": self.transcribe_enabled, "lyrics": self.lyrics_enabled}

    # ── ETA ───────────────────────────────────────────────────────────────────

    def _expected(self, job):
        """Whole-job seconds from priors/measurements, ignoring progress."""
        t = self.timing.get
        req = job.request
        if job.task in ("generate", "plan"):
            abc_needed = req.get("cot", "full") != "off" and req.get("abc") is None
            abc_max = (req.get("abc_sampling") or {}).get("max_tokens", 4096)
            abc_len = min(t("abc_len"), abc_max)
            seconds = t("overhead_s") * (0.5 if job.task == "plan" else 1.0)
            if abc_needed:
                seconds += abc_len / t("abc_tok_s")
            if job.task == "generate":
                sem_max = (req.get("semantic_sampling") or {}).get("max_tokens", 9000)
                sem_len = min(t("semantic_len"), sem_max)
                seconds += sem_len / t("semantic_tok_s") + sem_len * (t("nar_s_per_token") + t("vae_s_per_token"))
            return seconds
        if job.task == "decode":
            return t("overhead_s") + req.get("frames", t("semantic_len")) * t("vae_s_per_token") * 1.5
        if job.task == "transcribe":
            return t("transcribe_load_s") + req.get("audio_seconds", 240.0) * t("transcribe_s_per_audio_s")
        if job.task == "lyrics":
            return t("lyrics_load_s") + req.get("audio_seconds", 240.0) * t("lyrics_s_per_audio_s") * req.get("passes", 1)
        return 0.0

    def _remaining(self, job):
        if job.status != "running":
            return self._expected(job)
        t = self.timing.get
        req = job.request
        now = _now()
        in_phase = now - (job.phase_started or job.started or now)
        rate = lambda key: (job.tokens / in_phase) if job.tokens >= 50 and in_phase > 0 else t(key)
        if job.task in ("generate", "plan"):
            abc_max = (req.get("abc_sampling") or {}).get("max_tokens", 4096)
            sem_max = (req.get("semantic_sampling") or {}).get("max_tokens", 9000)
            sem_len = min(t("semantic_len"), sem_max)
            remaining = 0.0
            if job.phase == "planning":
                remaining += max(min(t("abc_len"), abc_max) - job.tokens, 0) / rate("abc_tok_s")
                if job.task == "generate":
                    remaining += sem_len / t("semantic_tok_s") + sem_len * (t("nar_s_per_token") + t("vae_s_per_token"))
            elif job.phase == "semantic":
                remaining += max(sem_len - job.tokens, 0) / rate("semantic_tok_s")
                remaining += max(sem_len, job.tokens) * (t("nar_s_per_token") + t("vae_s_per_token"))
            elif job.phase == "synthesizing":
                remaining += max(job.semantic_tokens * t("nar_s_per_token") - in_phase, 0)
                remaining += job.semantic_tokens * t("vae_s_per_token")
            elif job.phase == "decoding":
                remaining += max(job.semantic_tokens * t("vae_s_per_token") - in_phase, 0)
            return remaining + 3.0
        if job.task == "decode":
            return max(self._expected(job) - (now - job.started), 5.0)
        if job.task in ("transcribe", "lyrics"):
            prefix = job.task
            work = req.get("audio_seconds", 240.0) * t(f"{prefix}_s_per_audio_s") * (req.get("passes", 1) if job.task == "lyrics" else 1)
            if job.phase == "loading":
                return max(t(f"{prefix}_load_s") - in_phase, 2.0) + work
            return max(work - in_phase, 2.0)
        return 0.0

    def eta(self, job):
        if job.terminal:
            return 0
        with self.lock:
            ahead = []
            for jid in self.order:
                other = self.jobs[jid]
                if other.id == job.id:
                    break
                if other.status in ("queued", "running"):
                    ahead.append(other)
        total = self._remaining(job) + sum(self._remaining(o) for o in ahead)
        return int(round(total))

    # ── worker ────────────────────────────────────────────────────────────────

    def _worker(self):
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            with self.lock:
                job.status, job.started, self.running = "running", _now(), job.id
            try:
                runner = {"generate": self._run_generate, "plan": self._run_generate,
                          "decode": self._run_decode, "transcribe": self._run_transcribe,
                          "lyrics": self._run_lyrics}[job.task]
                runner(job)
                self._finish(job, "done", None)
            except InterruptedError as exc:
                self._finish(job, "cancelled", str(exc))
            except Exception as exc:  # noqa: BLE001 - every failure must land in the job, not the thread
                message = str(exc) if isinstance(exc, WorkerError) else f"{type(exc).__name__}: {exc}"
                if type(exc).__name__ == "OutOfMemoryError":
                    message = "CUDA out of memory: " + str(exc).splitlines()[0]
                    self._empty_cache()
                self._finish(job, "failed", message)
            finally:
                with self.lock:
                    self.running = None
                self._prune()

    def _phase(self, job, phase):
        with self.lock:
            job.phase, job.phase_started, job.tokens = phase, _now(), 0

    def _finish(self, job, status, error):
        with self.lock:
            if job.terminal:
                return
            job.status, job.error, job.finished, job.proc = status, error, _now(), None
            if status != "done":
                job.artifacts = None
        if status == "cancelled":
            shutil.rmtree(job.dir, ignore_errors=True)
        else:
            if status == "failed" and job.dir.is_dir():
                self._write_json(job.dir / "failure.json", {"status": "failed", "error": error, "request": job.request})
            self._write_meta(job)
        job.done_event.set()

    def _write_meta(self, job):
        if not job.dir.is_dir():
            return
        self._write_json(job.dir / "server_job.json", {
            "id": job.id, "task": job.task, "status": job.status, "request": job.request,
            "created": job.created, "started": job.started, "finished": job.finished,
            "error": job.error, "artifacts": job.artifacts, "result": job.result,
        })

    @staticmethod
    def _write_json(path, value):
        from yue2.storage import write_json
        write_json(path, value)

    def _empty_cache(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _offload_mot(self):
        """Move the 3B model off the GPU before another process needs it (transcription)."""
        model = getattr(self.pipe, "_model", None)
        if model is not None:
            try:
                model.to("cpu")
            except Exception:  # noqa: BLE001
                pass
        self._empty_cache()

    # ── tasks ─────────────────────────────────────────────────────────────────

    def _song_request(self, req):
        from yue2.protocol import SongRequest
        return SongRequest(style=req["style"], lyrics=req["lyrics"], cot=req.get("cot", "full"),
                           seed=req["seed"], abc=req.get("abc"), cfg_scale=req.get("cfg_scale"),
                           id=req.get("id", "song"))

    def _run_generate(self, job):
        from yue2.pipeline import SongResult
        from yue2.storage import identity, write_json
        pipe, req = self.pipe, job.request
        request = self._song_request(req)
        abc_sampling, semantic_sampling = req.get("abc_sampling"), req.get("semantic_sampling")

        def on_token(phase, token):
            with self.lock:
                job.tokens += 1
                if phase == "abc":
                    job.abc_tokens += 1
                else:
                    job.semantic_tokens += 1

        start = time.perf_counter()
        planning = request.cot != "off" and request.abc is None
        self._phase(job, "planning" if planning else "semantic")
        plan = pipe.plan(request=request, abc_sampling=abc_sampling, cancelled=job.cancel.is_set, on_token=on_token)
        abc_seconds = plan.timing.get("seconds", 0.0)
        if planning and job.abc_tokens and abc_seconds:
            self.timing.update(abc_tok_s=job.abc_tokens / abc_seconds, abc_len=float(job.abc_tokens))
        if job.task == "plan":
            self._phase(job, "saving")
            plan.save(job.dir)
            write_json(job.dir / "request.json", request.to_dict())
            write_json(job.dir / "provenance.json", {"weights": pipe.weights,
                       "config": pipe.effective_config(request, abc_sampling, semantic_sampling)})
            manifest = json.loads((job.dir / "plan_manifest.json").read_text())
            job.artifacts = sorted(manifest) + ["plan_manifest.json", "request.json", "provenance.json"]
            job.result = {"truncated": {"abc": plan.truncated}, "abc_tokens": job.abc_tokens,
                          "timing": {"abc": plan.timing, "e2e_seconds": time.perf_counter() - start}}
            return

        config = pipe.effective_config(request, abc_sampling, semantic_sampling)
        request_id = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
        self._phase(job, "semantic")
        semantic = pipe.generate_semantic(plan, sampling=semantic_sampling, cancelled=job.cancel.is_set,
                                          on_token=on_token)
        sem_seconds = semantic.timing.get("seconds", 0.0)
        if job.semantic_tokens and sem_seconds:
            self.timing.update(semantic_tok_s=job.semantic_tokens / sem_seconds, semantic_len=float(job.semantic_tokens))

        self._phase(job, "synthesizing")
        nar_start = time.perf_counter()
        latents = pipe.synthesize(semantic, cancelled=job.cancel.is_set)
        nar_seconds = time.perf_counter() - nar_start
        if job.cancel.is_set():
            raise InterruptedError("Cancelled before VAE")

        self._phase(job, "decoding")
        vae_start = time.perf_counter()
        audio = pipe.decode(latents)
        vae_seconds = time.perf_counter() - vae_start
        if job.cancel.is_set():
            raise InterruptedError("Cancelled after VAE")
        frames = max(len(semantic.tokens), 1)
        self.timing.update(nar_s_per_token=nar_seconds / frames, vae_s_per_token=vae_seconds / frames)

        timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
                  "vae_seconds": vae_seconds, "load": dict(pipe.load_timing),
                  "e2e_seconds": time.perf_counter() - start}
        song = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)
        self._phase(job, "saving")
        receipt = song.save_artifacts(job.dir)
        job.artifacts = sorted(receipt["artifacts"]) + ["result.json"] + self._mp3(job)
        job.result = {"truncated": receipt["truncated"], "audio_seconds": receipt["audio_seconds"],
                      "identity": receipt["identity"], "timing": timing}

    def _mp3(self, job):
        """Delivery copy, made once the manifests are final so it never enters result.json."""
        try:
            mp3 = convert_mp3(job.artifact_root / "audio.flac")
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"job {job.id}: mp3 conversion failed: {exc}", flush=True)
            return []
        return [mp3.name] if mp3 else []

    def _legacy_weights(self):
        from yue2.storage import model_identity
        if self._legacy_identity is None:
            self._legacy_identity = model_identity(self.legacy_vae)
        return self._legacy_identity

    def _run_decode(self, job):
        from yue2.pipeline import SemanticResult, SongResult, SymbolicPlan
        from yue2.storage import identity, sha256_file, verify_result, write_json
        pipe, req = self.pipe, job.request
        source = self.jobs[req["source_job"]]
        src = source.artifact_root
        original = verify_result(src)
        original_config = json.loads((src / "config.json").read_text())
        plan = SymbolicPlan.load(src)
        tokens = np.load(src / "semantic.npy", allow_pickle=False)
        latents = np.load(src / "latent.npy", allow_pickle=False)
        semantic = SemanticResult(plan, tokens.tolist(), original["timing"].get("semantic", {}),
                                  original["truncated"]["semantic"])
        legacy = req.get("vae") == "legacy"
        vae_dir = Path(self.legacy_vae) if legacy else pipe.vae_dir
        weights = {"mot": pipe.weights["mot"], "vae": self._legacy_weights() if legacy else pipe.weights["vae"]}

        write_json(job.dir / "source_generation.json", {
            "source_job": source.id, "source_result_sha256": sha256_file(src / "result.json"),
            "source_latent_sha256": sha256_file(src / "latent.npy"),
            "source_semantic_sha256": sha256_file(src / "semantic.npy"),
            "config": original_config, "identity": original["identity"], "weights": original["weights"],
        })
        self._phase(job, "decoding")
        start = time.perf_counter()
        audio = pipe.decode(latents, vae=str(vae_dir) if legacy else None)
        elapsed = time.perf_counter() - start
        if job.cancel.is_set():
            raise InterruptedError("Cancelled after VAE")
        current = pipe.effective_config(plan.request)
        config = dict(original_config)
        for key in ("vae_dtype", "vae_decode", "vae_core_frames", "vae_halo_frames"):
            config[key] = current[key]
        config["decoder_release"] = json.loads((vae_dir / "config.json").read_text()).get("release_variant")
        config["cached_decode"] = {"source_identity": original["identity"], "source_job": source.id,
                                   "source_config_sha256": sha256_file(src / "config.json"),
                                   "runtime_sha256": pipe.runtime_sha256, "device": str(pipe.device),
                                   "memory_budget_gib": pipe.memory_budget_gib, "decoder": req.get("vae", "standard")}
        timing = {"operation": "decode_cached_latents", "vae_seconds": elapsed,
                  "semantic": semantic.timing, "source_generation": original["timing"]}
        digest = identity({"request": plan.request.to_dict(), "config": config, "weights": weights})
        song = SongResult(audio, 48000, semantic, latents, config, weights, timing, digest)
        self._phase(job, "saving")
        receipt = song.save_artifacts(job.dir)
        job.artifacts = sorted(receipt["artifacts"]) + ["result.json"] + self._mp3(job)
        job.result = {"truncated": receipt["truncated"], "audio_seconds": receipt["audio_seconds"],
                      "identity": receipt["identity"], "timing": timing}

    def _run_transcribe(self, job):
        req = job.request
        paths = {name: resolve(name) for name in TRANSCRIPTION}
        command = [*self.worker_command, str(self.worker_script), str(self._input_audio(job)),
                   "--output", str(job.artifact_root), "--task", req["task"], "--preset", req["preset"],
                   "--dtype", req["dtype"], "--offline", "--device", "cuda",
                   "--model", str(paths["SheetSage2"]), "--base-model", str(paths["MERT-v2-FullSong"])]
        if req.get("max_seconds"):
            command += ["--max-seconds", str(req["max_seconds"])]
        manifest = self._run_worker(job, command, "transcription_manifest.json", "transcribe")
        job.result = {"warnings": manifest.get("warnings", []), "timing": job.result["timing"]}

    def _run_lyrics(self, job):
        req = job.request
        model = resolve(LYRICS[0])
        command = [*self.worker_command, str(self.lyrics_worker), str(self._input_audio(job)),
                   "--output", str(job.artifact_root), "--model", str(model), "--language", req.get("language", "auto"),
                   "--passes", str(req.get("passes", 1)), "--offline", "--device", "cuda:0"]
        reference = job.dir / "reference_lyrics.txt"
        if req.get("lyrics"):
            reference.write_text(req["lyrics"], encoding="utf-8")
            command += ["--lyrics-file", str(reference)]
        self._run_worker(job, command, "lyrics_manifest.json", "lyrics")
        report = json.loads((job.artifact_root / "lyrics_asr.json").read_text())
        job.result = {"per": report.get("per"), "unit_error_rate": report.get("unit_error_rate"),
                      "language_detected": report.get("language_detected"), "transcript": report["passes"][report["best_pass"] - 1]["text"],
                      "audio_seconds": report.get("audio_seconds"), "timing": job.result["timing"]}

    def _input_audio(self, job):
        """The uploaded file, or the source job's audio.flac for jobs submitted with source_job."""
        req = job.request
        if req.get("source_job"):
            source = self.jobs[req["source_job"]]
            return source.artifact_root / "audio.flac"
        return job.dir / "upload" / req["filename"]

    def _run_worker(self, job, command, manifest_name, timing_prefix):
        """Run a worker subprocess in its own uv environment; return its manifest on success."""
        out = job.artifact_root
        if out.exists():
            shutil.rmtree(out)
        self._offload_mot()
        self._phase(job, "loading")
        log = (job.dir / "worker.log").open("w")
        start = time.perf_counter()
        loaded_at = None
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=offline_env(), cwd=str(HERE))
            with self.lock:
                job.proc = proc
            while proc.poll() is None:
                if job.cancel.is_set():
                    proc.terminate()
                    try:
                        proc.wait(10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise InterruptedError(f"Cancelled during {job.task}")
                if loaded_at is None and (out / "model_provenance.json").is_file():
                    loaded_at = time.perf_counter()
                    self._phase(job, "transcribing" if job.task == "transcribe" else "recognizing")
                time.sleep(0.5)
        finally:
            log.close()
            with self.lock:
                job.proc = None
            shutil.rmtree(job.dir / "upload", ignore_errors=True)
        if job.cancel.is_set():
            # cancel() may have terminated the process before the loop saw the flag.
            raise InterruptedError(f"Cancelled during {job.task}")
        manifest_path = out / manifest_name
        if proc.returncode == 0 and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") == "complete":
                total = time.perf_counter() - start
                if loaded_at is not None:
                    self.timing.update(**{f"{timing_prefix}_load_s": loaded_at - start})
                    audio_seconds = job.request.get("audio_seconds")
                    if audio_seconds:
                        passes = job.request.get("passes", 1) if job.task == "lyrics" else 1
                        self.timing.update(**{f"{timing_prefix}_s_per_audio_s": (time.perf_counter() - loaded_at) / (audio_seconds * passes)})
                job.artifacts = sorted(manifest.get("artifacts", {})) + [manifest_name]
                job.result = {"timing": {"e2e_seconds": total, "load_seconds": (loaded_at - start) if loaded_at else None}}
                return manifest
        error = None
        failure = out / "failure.json"
        if failure.is_file():
            try:
                error = json.loads(failure.read_text()).get("error")
            except ValueError:
                error = None
        if not error:
            lines = [line for line in (job.dir / "worker.log").read_text(errors="replace").splitlines() if line.strip()]
            error = lines[-1] if lines else f"worker exited with {proc.returncode}"
        raise WorkerError(error)

    # ── restart recovery and retention ────────────────────────────────────────

    def _recover(self):
        for path in sorted(self.out_dir.iterdir() if self.out_dir.is_dir() else []):
            meta = path / "server_job.json"
            if not meta.is_file():
                continue
            try:
                data = json.loads(meta.read_text())
            except ValueError:
                continue
            if data.get("status") != "done" or not data.get("artifacts"):
                continue
            job = Job(data["id"], data["task"], data.get("request", {}), path, created=data.get("created", 0.0),
                      status="done", started=data.get("started"), finished=data.get("finished"),
                      artifacts=data["artifacts"], result=data.get("result"), recovered=True)
            if not all((job.artifact_root / name).is_file() for name in job.artifacts):
                continue
            job.done_event.set()
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.order.sort(key=lambda jid: self.jobs[jid].created)

    def _prune(self):
        if not self.keep_jobs:
            return
        with self.lock:
            finished = [self.jobs[j] for j in self.order if self.jobs[j].terminal]
            excess = finished[:-self.keep_jobs] if len(finished) > self.keep_jobs else []
            for job in excess:
                self.order.remove(job.id)
                del self.jobs[job.id]
        for job in excess:
            shutil.rmtree(job.dir, ignore_errors=True)


def audio_duration(path):
    """Seconds of audio, for the ETA only; None when the container is not readable here."""
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:  # noqa: BLE001
        return None


def convert_mp3(flac, mp3=None, quality=MP3_QUALITY):
    """Encode audio.flac to audio.mp3 next to it with ffmpeg. Returns the path, or None without ffmpeg.

    The FLAC stays the archival, hash-verified artifact; the MP3 is the delivery copy.
    """
    flac = Path(flac)
    mp3 = Path(mp3) if mp3 else flac.with_suffix(".mp3")
    if mp3.is_file() and mp3.stat().st_mtime >= flac.stat().st_mtime:
        return mp3
    if not shutil.which("ffmpeg"):
        return None
    tmp = mp3.with_name(mp3.name + ".tmp.mp3")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(flac), "-codec:a", "libmp3lame",
                    "-q:a", quality, str(tmp)], check=True, capture_output=True)
    tmp.replace(mp3)
    return mp3
