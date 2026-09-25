"""HTTP contract the skill's run_yue2.py / transcribe.py rely on, against the fake pipeline."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import FAKE_WORKER, FakePipeline, wait_done, wait_until

REQUEST = {"style": "English, warm piano pop, 88 BPM", "lyrics": "[Verse]\nNeon fades\n[Chorus]\nLet the day\n",
           "cot": "full", "seed": 831001, "id": "city_lights"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["running"] is None and body["queued"] == 0
    assert body["transcribe"] is True and body["decoders"] == ["standard"]


def test_generate_roundtrip(client, jq):
    r = client.post("/jobs", json=REQUEST)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "queued" and body["seed"] == 831001 and body["eta_s"] > 0
    view = wait_done(client, body["id"])
    assert view["status"] == "done", view
    assert view["abc_tokens"] == 12 and view["semantic_tokens"] == 40
    assert view["truncated"] == {"abc": False, "semantic": False}
    assert view["audio_seconds"] == pytest.approx(40 * 0.04)
    assert set(view["artifacts"]) >= {"result.json", "audio.flac", "score.abc", "plan.json", "plan_manifest.json",
                                      "semantic.npy", "latent.npy", "request.json", "config.json", "prefix.npy",
                                      "abc_tokens.npy"}
    assert "server_job.json" not in view["artifacts"]
    # Every listed artifact downloads unchanged, and result.json's manifest matches what we got.
    result = client.get(f"/jobs/{body['id']}/artifacts/result.json").json()
    assert result["status"] == "complete" and result["identity"] == view["identity"]
    for name, entry in result["artifacts"].items():
        data = client.get(f"/jobs/{body['id']}/artifacts/{name}").content
        assert sha(data) == entry["sha256"] and len(data) == entry["bytes"], name
    assert client.get(f"/jobs/{body['id']}/score").text.startswith("X:1")
    audio = client.get(f"/jobs/{body['id']}/audio", params={"format": "flac"})
    assert audio.headers["content-type"].startswith("audio/flac")
    data, rate = sf.read(io.BytesIO(audio.content))
    assert rate == 48000 and data.shape == (40 * 1920, 2)
    if shutil.which("ffmpeg"):
        # MP3 is the default delivery copy, made from the FLAC after the manifests are written.
        assert "audio.mp3" in view["artifacts"] and view["audio_url"].endswith("/audio")
        mp3 = client.get(f"/jobs/{body['id']}/audio")
        assert mp3.headers["content-type"] == "audio/mpeg" and mp3.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
        assert client.get(f"/jobs/{body['id']}/artifacts/audio.mp3").content == mp3.content
        decoded, rate = sf.read(io.BytesIO(mp3.content))
        assert rate == 48000 and abs(len(decoded) - 40 * 1920) < 4000      # encoder padding only
        assert "audio.mp3" not in result["artifacts"]                       # not part of YuE2's manifest
    wav = client.get(f"/jobs/{body['id']}/audio", params={"format": "wav"})
    assert wav.headers["content-type"].startswith("audio/wav")
    # The directory is byte-for-byte what YuE2's own verifier expects.
    from yue2.storage import verify_result
    verify_result(jq.get(body["id"]).dir)
    assert client.get("/health").json()["eta_source"] == "measured"
    assert client.get("/jobs").json()[0]["id"] == body["id"]
    # Unlisted names never reach the filesystem.
    assert client.get(f"/jobs/{body['id']}/artifacts/server_job.json").status_code == 404
    assert client.get(f"/jobs/{body['id']}/artifacts/../../etc/passwd").status_code == 404


def test_cot_off_has_no_score(client):
    r = client.post("/jobs", json={**REQUEST, "cot": "off"})
    view = wait_done(client, r.json()["id"])
    assert view["status"] == "done" and view["abc_tokens"] == 0 and view["score_url"] is None
    assert client.get(f"/jobs/{view['id']}/score").status_code == 404


def test_supplied_abc_skips_planning(client, pipe):
    r = client.post("/jobs", json={**REQUEST, "cot": "melody", "abc": "X:1\nK:C\nV:Vocal\nC4 D4|\n"})
    view = wait_done(client, r.json()["id"])
    assert view["status"] == "done" and view["abc_tokens"] == 0
    assert client.get(f"/jobs/{view['id']}/score").text.startswith("X:1\nK:C")


def test_plan_task(client, jq):
    r = client.post("/jobs", json={**REQUEST, "task": "plan"})
    assert r.status_code == 202
    view = wait_done(client, r.json()["id"])
    assert view["status"] == "done" and view["truncated"] == {"abc": False}
    assert view["audio_url"] is None and view["score_url"].endswith("/score")
    assert {"plan.json", "plan_manifest.json", "abc_tokens.npy", "prefix.npy", "score.abc"} <= set(view["artifacts"])
    from yue2.pipeline import SymbolicPlan
    plan = SymbolicPlan.load(jq.get(view["id"]).dir)
    assert plan.abc.startswith("X:1")
    assert client.get(f"/jobs/{view['id']}/audio").status_code == 404


@pytest.mark.parametrize("body, message", [
    ({"style": "x"}, "required"),
    ({**REQUEST, "cot": "off", "abc": "X:1"}, "External ABC"),
    ({**REQUEST, "task": "plan", "cot": "off"}, "cot=off has no score"),
    ({**REQUEST, "semantic_sampling": {"top_p": 5}}, "top_p"),
    ({**REQUEST, "seed": -1}, "seed"),
    ({**REQUEST, "id": "../x"}, "filename-safe"),
    ({**REQUEST, "style": "a", "tags": "b"}, "aliases"),
    ({**REQUEST, "bogus": 1}, "bogus"),
    ({**REQUEST, "task": "decode"}, "source_job"),
    ({**REQUEST, "task": "decode", "source_job": "nope"}, "finished generate"),
])
def test_validation(client, body, message):
    r = client.post("/jobs", json=body)
    assert r.status_code == 422, r.text
    assert message in r.text


def test_random_seed_when_omitted(client):
    body = {k: v for k, v in REQUEST.items() if k != "seed"}
    a, b = client.post("/jobs", json=body).json(), client.post("/jobs", json=body).json()
    assert isinstance(a["seed"], int) and a["seed"] != b["seed"]


def test_cancel_queued_and_running(tmp_path, fake_models):
    from fastapi.testclient import TestClient
    from yue2_jobs import JobQueue
    from yue2_server import create_app
    slow = FakePipeline(tmp_path, semantic_tokens=400, token_delay=0.01)
    jq = JobQueue(slow, out_dir=tmp_path / "out", gpu="fake", timing_path=tmp_path / "t.json",
                  worker=FAKE_WORKER, worker_command=[sys.executable])
    with TestClient(create_app(jq, {"static": {}, "max_upload_mb": 1})) as client:
        first = client.post("/jobs", json=REQUEST).json()
        second = client.post("/jobs", json=REQUEST).json()
        assert second["position"] == 1          # first is already running; position counts queued jobs
        assert client.delete(f"/jobs/{second['id']}").json()["status"] == "cancelled"
        assert wait_until(lambda: client.get(f"/jobs/{first['id']}").json()["phase"] == "semantic")
        assert client.get("/health").json()["running"] == first["id"]
        r = client.delete(f"/jobs/{first['id']}")
        assert r.json()["cancel_pending"] is True
        view = wait_done(client, first["id"])
        assert view["status"] == "cancelled" and view["artifacts"] is None
        assert not (tmp_path / "out" / first["id"]).exists()
        assert client.get(f"/jobs/{first['id']}/audio").status_code == 409
        assert client.delete(f"/jobs/{first['id']}").json()["status"] == "cancelled"  # idempotent
        third = client.post("/jobs", json={**REQUEST, "cot": "off"}).json()
        assert wait_done(client, third["id"])["status"] == "done"       # the worker survived the cancel


def test_failure_is_reported(tmp_path, fake_models):
    from fastapi.testclient import TestClient
    from yue2_jobs import JobQueue
    from yue2_server import create_app
    broken = FakePipeline(tmp_path)
    broken.synthesize = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    jq = JobQueue(broken, out_dir=tmp_path / "out", gpu="fake", timing_path=tmp_path / "t.json")
    with TestClient(create_app(jq, {"static": {}, "max_upload_mb": 1})) as client:
        job = client.post("/jobs", json=REQUEST).json()
        view = wait_done(client, job["id"])
        assert view["status"] == "failed" and view["error"] == "RuntimeError: boom"
        assert json.loads((tmp_path / "out" / job["id"] / "failure.json").read_text())["error"] == "RuntimeError: boom"
        assert client.get("/health").json()["running"] is None


def test_decode_reuses_latents(client, jq, pipe, fake_models):
    src = wait_done(client, client.post("/jobs", json=REQUEST).json()["id"])
    r = client.post("/jobs", json={"task": "decode", "source_job": src["id"], "vae": "legacy"})
    assert r.status_code == 422 and "legacy decoder" in r.text
    jq.legacy_vae = fake_models["YuE2-Vae-legacy"]
    jq._legacy_identity = {"files": {}, "config_sha256": "f" * 64}
    r = client.post("/jobs", json={"task": "decode", "source_job": src["id"], "vae": "legacy"})
    assert r.status_code == 202, r.text
    view = wait_done(client, r.json()["id"])
    assert view["status"] == "done" and view["identity"] != src["identity"]
    assert pipe.calls[-1] == "decode:legacy"
    a = client.get(f"/jobs/{src['id']}/artifacts/latent.npy").content
    b = client.get(f"/jobs/{view['id']}/artifacts/latent.npy").content
    assert a == b
    config = client.get(f"/jobs/{view['id']}/artifacts/config.json").json()
    assert config["decoder_release"] == "legacy" and config["cached_decode"]["source_job"] == src["id"]
    assert "source_generation.json" in view["artifacts"]
    from yue2.storage import verify_result
    verify_result(jq.get(view["id"]).dir)


def _wav_bytes(seconds=1.0):
    t = np.linspace(0, seconds, int(48000 * seconds), endpoint=False, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([np.sin(440 * t)] * 2, axis=1), 48000, format="WAV")
    return buffer.getvalue()


def test_transcribe_roundtrip(client, jq, pipe):
    wav = _wav_bytes()
    r = client.post("/jobs/transcribe", files={"audio": ("song.wav", wav, "audio/wav")},
                    data={"task": "melody-full"})
    assert r.status_code == 202, r.text
    view = wait_done(client, r.json()["id"])
    assert view["status"] == "done", view
    assert view["request"]["audio_seconds"] == 1.0 and view["warnings"] == ["fake"]
    assert set(view["artifacts"]) == {"model_provenance.json", "score.abc", "melody.mid", "events.json",
                                      "transcription_manifest.json"}
    manifest = client.get(f"/jobs/{view['id']}/artifacts/transcription_manifest.json").json()
    assert manifest["source_audio_sha256"] == sha(wav)
    assert client.get(f"/jobs/{view['id']}/score").text.startswith("X:1")
    assert client.get(f"/jobs/{view['id']}/audio").status_code == 404
    assert not (jq.get(view["id"]).dir / "upload").exists()


def test_transcribe_rejections(client, monkeypatch):
    wav = _wav_bytes()
    assert client.post("/jobs/transcribe", files={"audio": ("song.txt", b"x", "text/plain")}).status_code == 415
    assert client.post("/jobs/transcribe", files={"audio": ("s.wav", wav, "audio/wav")}, data={"task": "x"}).status_code == 422
    big = client.post("/jobs/transcribe", files={"audio": ("big.wav", _wav_bytes(6.0), "audio/wav")})
    assert big.status_code == 413
    monkeypatch.setenv("FAKE_WORKER_FAIL", "abc")
    view = wait_done(client, client.post("/jobs/transcribe", files={"audio": ("s.wav", wav, "audio/wav")}).json()["id"])
    assert view["status"] == "failed" and view["error"] == "no usable ABC"


def test_transcribe_cancel(tmp_path, fake_models, monkeypatch):
    from fastapi.testclient import TestClient
    from yue2_jobs import JobQueue
    from yue2_server import create_app
    monkeypatch.setenv("FAKE_WORKER_SLEEP", "30")
    jq = JobQueue(FakePipeline(tmp_path), out_dir=tmp_path / "out", gpu="fake", timing_path=tmp_path / "t.json",
                  worker=FAKE_WORKER, worker_command=[sys.executable])
    with TestClient(create_app(jq, {"static": {}, "max_upload_mb": 1})) as client:
        job = client.post("/jobs/transcribe", files={"audio": ("s.wav", _wav_bytes(), "audio/wav")}).json()
        assert wait_until(lambda: client.get(f"/jobs/{job['id']}").json()["phase"] == "transcribing", timeout=15)
        started = time.time()
        client.delete(f"/jobs/{job['id']}")
        view = wait_done(client, job["id"])
        assert view["status"] == "cancelled" and time.time() - started < 15


def test_recovery_after_restart(tmp_path, pipe, fake_models):
    from fastapi.testclient import TestClient
    from yue2_jobs import JobQueue
    from yue2_server import create_app
    out = tmp_path / "out"
    jq = JobQueue(pipe, out_dir=out, gpu="fake", timing_path=tmp_path / "t.json")
    with TestClient(create_app(jq, {"static": {}, "max_upload_mb": 1})) as client:
        done = wait_done(client, client.post("/jobs", json=REQUEST).json()["id"])
        failed_id = client.post("/jobs", json={**REQUEST, "task": "plan"}).json()["id"]
        wait_done(client, failed_id)
    # A directory whose files vanished is not recovered.
    shutil.rmtree(out / failed_id)
    jq2 = JobQueue(FakePipeline(tmp_path / "second"), out_dir=out, gpu="fake", timing_path=tmp_path / "t.json",
                   start_worker=False)
    assert set(jq2.jobs) == {done["id"]}
    with TestClient(create_app(jq2, {"static": {}, "max_upload_mb": 1})) as client:
        view = client.get(f"/jobs/{done['id']}").json()
        assert view["status"] == "done" and view["recovered"] is True and view["identity"] == done["identity"]
        assert client.get(f"/jobs/{done['id']}/audio").status_code == 200
        assert jq2.timing.source == "measured"     # rates persisted across the restart


def test_keep_jobs_prunes_oldest(tmp_path, pipe, fake_models):
    from fastapi.testclient import TestClient
    from yue2_jobs import JobQueue
    from yue2_server import create_app
    jq = JobQueue(pipe, out_dir=tmp_path / "out", gpu="fake", timing_path=tmp_path / "t.json", keep_jobs=2)
    with TestClient(create_app(jq, {"static": {}, "max_upload_mb": 1})) as client:
        ids = [wait_done(client, client.post("/jobs", json={**REQUEST, "cot": "off"}).json()["id"])["id"] for _ in range(3)]
        assert wait_until(lambda: client.get(f"/jobs/{ids[0]}").status_code == 404)
        assert [j["id"] for j in client.get("/jobs").json()] == [ids[2], ids[1]]
        assert not (tmp_path / "out" / ids[0]).exists()


def test_first_measurement_replaces_prior(tmp_path):
    from yue2_jobs import PRIORS, Timing
    timing = Timing("fake", tmp_path / "t.json", profile={"generate": {"semantic_tok_s": 120.0}})
    assert timing.source == "dryrun" and timing.get("semantic_tok_s") == 120.0
    timing.update(transcribe_load_s=9.0, semantic_tok_s=150.0)
    assert timing.get("transcribe_load_s") == 9.0 and timing.get("semantic_tok_s") == 150.0
    timing.update(transcribe_load_s=19.0)
    assert timing.get("transcribe_load_s") == pytest.approx(12.0)      # 0.7 * 9 + 0.3 * 19
    assert timing.get("abc_tok_s") == PRIORS["abc_tok_s"]
    reloaded = Timing("fake", tmp_path / "t.json")
    assert reloaded.source == "measured" and reloaded.measured >= {"transcribe_load_s", "semantic_tok_s"}
    reloaded.update(transcribe_load_s=100.0)
    assert reloaded.get("transcribe_load_s") == pytest.approx(0.7 * 12.0 + 0.3 * 100.0)
