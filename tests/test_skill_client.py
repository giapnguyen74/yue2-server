"""The skill's stdlib clients against a live uvicorn instance serving the fake pipeline."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import FAKE_WORKER, FakePipeline

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "yue2-music-server"
SCRIPTS = SKILL / "scripts"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path, fake_models):
    import uvicorn
    from yue2_jobs import JobQueue
    from yue2_server import create_app
    jq = JobQueue(FakePipeline(tmp_path), out_dir=tmp_path / "out", gpu="fake", timing_path=tmp_path / "t.json",
                  worker=FAKE_WORKER, worker_command=[sys.executable])
    app = create_app(jq, {"static": {"gpu": "fake"}, "max_upload_mb": 5})
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started
    yield f"http://127.0.0.1:{port}", jq
    srv.should_exit = True
    thread.join(5)


def run_script(name, *args, env=None):
    proc = subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)], cwd=str(SCRIPTS),
                          capture_output=True, text=True, env={**os.environ, **(env or {})})
    return proc


def test_run_yue2_generate_plan_decode(server, tmp_path):
    url, jq = server
    env = {"YUE2_SERVER": url}
    out = tmp_path / "song"
    proc = run_script("run_yue2.py", "generate", "--request", SKILL / "assets" / "prompt.json", "--output", out, env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    run = json.loads((out / "run.json").read_text())["results"][0]
    assert run["status"] == "complete" and run["mode"] == "full"
    for name in ("audio.flac", "score.abc", "result.json", "latent.npy", "abc_check.json", "job.json", "invocation.json"):
        assert (out / name).is_file(), name
    assert json.loads((out / "abc_check.json").read_text())["status"] == "passed"
    assert json.loads((out / "invocation.json").read_text())["health"]["gpu"] == "fake"

    plan_out = tmp_path / "plan"
    proc = run_script("run_yue2.py", "plan", "--request", SKILL / "assets" / "prompt.json", "--output", plan_out, env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (plan_out / "score.abc").is_file() and (plan_out / "plan_manifest.json").is_file()

    # Edited score back in as ABC input, melody mode via the CLI flag.
    edited = tmp_path / "edited.abc"
    proc = run_script("abc_tools.py", "strip-chords", plan_out / "score.abc", edited)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    proc = run_script("run_yue2.py", "generate", "--request", SKILL / "assets" / "prompt.json", "--cot", "melody",
                      "--abc-file", edited, "--output", tmp_path / "edited", env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads((tmp_path / "edited" / "run.json").read_text())["results"][0]["status"] == "complete"

    # decode: legacy decoder not installed -> clean failure with the server's message.
    proc = run_script("run_yue2.py", "decode", "--source", out, "--output", tmp_path / "decoded", "--vae", "legacy", env=env)
    assert proc.returncode == 2 and "legacy decoder" in proc.stderr
    proc = run_script("run_yue2.py", "decode", "--source", out, "--output", tmp_path / "decoded2", "--vae", "standard", env=env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (tmp_path / "decoded2" / "audio.flac").is_file() and (tmp_path / "decoded2" / "source_generation.json").is_file()


def test_run_yue2_all_modes_and_unreachable(server, tmp_path):
    url, _ = server
    proc = run_script("run_yue2.py", "all-modes", "--request", SKILL / "assets" / "prompt.json",
                      "--output", tmp_path / "modes", "--quiet", env={"YUE2_SERVER": url})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    results = json.loads((tmp_path / "modes" / "run.json").read_text())["results"]
    assert [r["mode"] for r in results] == ["full", "melody", "off"] and all(r["status"] == "complete" for r in results)
    assert not (tmp_path / "modes" / "off" / "score.abc").exists()
    proc = run_script("run_yue2.py", "generate", "--request", SKILL / "assets" / "prompt.json",
                      "--output", tmp_path / "x", "--server", "http://127.0.0.1:9")
    assert proc.returncode == 2 and "cannot reach" in proc.stderr


def test_transcribe_client(server, tmp_path):
    url, _ = server
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    wav = tmp_path / "source.wav"
    sf.write(str(wav), np.stack([np.sin(440 * t)] * 2, axis=1), 48000)
    out = tmp_path / "transcription"
    proc = run_script("transcribe.py", wav, "--task", "melody-full", "--output", out, env={"YUE2_SERVER": url})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out / "score.abc").read_text().startswith("X:1")
    assert json.loads((out / "abc_check.json").read_text())["status"] == "passed"
    assert json.loads((out / "job.json").read_text())["task"] == "transcribe"
    # strip-chords on the result, then feed it back as a melody cover through the same server.
    proc = run_script("abc_tools.py", "strip-chords", out / "score.abc", tmp_path / "cover.abc")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    proc = run_script("run_yue2.py", "generate", "--request", SKILL / "assets" / "prompt.json", "--cot", "melody",
                      "--abc-file", tmp_path / "cover.abc", "--output", tmp_path / "cover", env={"YUE2_SERVER": url})
    assert proc.returncode == 0, proc.stderr + proc.stdout
