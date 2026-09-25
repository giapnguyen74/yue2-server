"""A fake YuE2Pipeline with the real artifact classes, so the queue and API run without a GPU."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yue2.pipeline import SemanticResult, SymbolicPlan  # noqa: E402
from yue2.protocol import CONTEXT, PROTOCOL_VERSION, GenerationConfig  # noqa: E402

# Real native two-voice scores from the YuE repository examples, so the skill's ABC checks pass.
FIXTURES = ROOT / "tests" / "fixtures"
FAKE_ABC = (FIXTURES / "score.abc").read_text(encoding="utf-8")
FAKE_MELODY_ABC = (FIXTURES / "melody.abc").read_text(encoding="utf-8")


class FakePipeline:
    """Same call surface as YuE2Pipeline for the parts the server uses. Deterministic and fast."""

    def __init__(self, tmp_path, abc_tokens=12, semantic_tokens=40, token_delay=0.0):
        self.abc_tokens, self.semantic_tokens, self.token_delay = abc_tokens, semantic_tokens, token_delay
        self.vae_dir = tmp_path / "fake-vae"
        self.vae_dir.mkdir(parents=True)
        (self.vae_dir / "config.json").write_text(json.dumps({"release_variant": "standard"}))
        self.weights = {"mot": {"files": {"model.safetensors": {"sha256": "a" * 64, "bytes": 1}}, "config_sha256": "b" * 64},
                        "vae": {"files": {"model.safetensors": {"sha256": "c" * 64, "bytes": 1}}, "config_sha256": "d" * 64}}
        self.load_timing = {}
        self.runtime_sha256 = "e" * 64
        self.device = "cuda:0"
        self.memory_budget_gib = 28.0
        self.backend, self.quantization, self.offload_ar, self.vae_core_frames = "torch", "none", False, 1024
        self.generation_config = GenerationConfig()
        self._model = None
        self.closed = False
        self.calls = []

    def _emit(self, phase, count, cancelled, on_token):
        for i in range(count):
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled during {phase}")
            if self.token_delay:
                time.sleep(self.token_delay)
            if on_token is not None:
                on_token(phase, 1000 + i)

    def plan(self, request=None, abc_sampling=None, cancelled=None, on_token=None, **kwargs):
        self.calls.append("plan")
        if request.cot == "off":
            return SymbolicPlan(request, None, [], [151643, 1, 2])
        if request.abc is not None:
            ids = [200 + (ord(c) % 50) for c in request.abc[:20]]
            return SymbolicPlan(request, request.abc, ids, [151643, 1] + ids, {"seconds": 0.0, "output_tokens": 0})
        t0 = time.perf_counter()
        self._emit("abc", self.abc_tokens, cancelled, on_token)
        abc = FAKE_MELODY_ABC if request.cot == "melody" else FAKE_ABC
        ids = list(range(300, 300 + self.abc_tokens))
        return SymbolicPlan(request, abc, ids, [151643, 1] + ids,
                            {"seconds": time.perf_counter() - t0 + 0.01, "output_tokens": self.abc_tokens}, False)

    def generate_semantic(self, plan, sampling=None, cancelled=None, on_token=None):
        self.calls.append("semantic")
        t0 = time.perf_counter()
        self._emit("semantic", self.semantic_tokens, cancelled, on_token)
        tokens = [(i * 7) % 32768 for i in range(self.semantic_tokens)]
        return SemanticResult(plan, tokens, {"seconds": time.perf_counter() - t0 + 0.01,
                                             "output_tokens": self.semantic_tokens}, False)

    def synthesize(self, semantic, cancelled=None):
        self.calls.append("synthesize")
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        rng = np.random.default_rng(semantic.plan.request.seed)
        return rng.standard_normal((len(semantic.tokens), 64)).astype(np.float32)

    def decode(self, latents, *, full=False, vae=None):
        self.calls.append(f"decode:{'legacy' if vae else 'standard'}")
        frames = np.asarray(latents).shape[0]
        t = np.linspace(0, frames * 0.04, frames * 1920, endpoint=False, dtype=np.float32)
        tone = 0.2 * np.sin(2 * np.pi * 220 * t)
        return np.stack([tone, tone], axis=1)

    def effective_config(self, request, abc_sampling=None, semantic_sampling=None):
        return {"generation": self.generation_config.to_dict(), "overrides": {}, "cot": request.cot,
                "cfg_scale": request.guidance, "backend": "torch", "quantization": "none",
                "model_dtype": "bfloat16", "vae_dtype": "float32", "vae_decode": "halo_crop",
                "vae_core_frames": 1024, "vae_halo_frames": 16, "device": self.device,
                "memory_budget_gib": self.memory_budget_gib, "offload_ar": False,
                "runtime_sha256": self.runtime_sha256, "decoder_release": "standard",
                "validation_status": "unvalidated", "context": CONTEXT, "version": PROTOCOL_VERSION}

    def close(self):
        self.closed = True


FAKE_WORKER = ROOT / "tests" / "fake_transcribe_worker.py"


@pytest.fixture
def fake_models(monkeypatch, tmp_path):
    """resolve() answers with empty directories so no Hugging Face cache is needed."""
    import yue2_jobs
    paths = {}
    for name in ("SheetSage2", "MERT-v2-FullSong", "YuE2-Vae-legacy"):
        path = tmp_path / "models" / name
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({"release_variant": "legacy" if "legacy" in name else None}))
        paths[name] = path
    monkeypatch.setattr(yue2_jobs, "resolve", lambda name: paths[name])
    return paths


@pytest.fixture
def pipe(tmp_path):
    return FakePipeline(tmp_path)


@pytest.fixture
def jq(tmp_path, pipe, fake_models):
    from yue2_jobs import JobQueue
    return JobQueue(pipe, out_dir=tmp_path / "outputs", gpu="fake", timing_path=tmp_path / "timing.json",
                    worker=FAKE_WORKER, worker_command=[sys.executable], transcribe=True, legacy_vae=None)


@pytest.fixture
def client(jq):
    from fastapi.testclient import TestClient
    from yue2_server import create_app
    app = create_app(jq, {"static": {"gpu": "fake", "vae": "standard"}, "max_upload_mb": 1})
    with TestClient(app) as c:
        yield c


def wait_done(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get(f"/jobs/{job_id}/wait", params={"timeout": 5}).json()
        if view["status"] in ("done", "failed", "cancelled"):
            return view
    raise AssertionError(f"job {job_id} did not finish: {view}")


def wait_until(predicate, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


__all__ = ["FakePipeline", "wait_done", "wait_until", "threading"]
