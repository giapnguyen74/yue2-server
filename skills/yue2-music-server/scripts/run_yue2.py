#!/usr/bin/env python3
"""Generate, plan, or re-decode through a running yue2-server; preserve native artifacts.

Standard library only. The server (see the repository's PLAN.md) owns the GPU and the
model snapshots; this helper submits one job per request, waits for it, downloads the
artifacts YuE2 itself wrote (result.json, score.abc, audio.flac, latent.npy, ...) and
checks their SHA-256 against the server's manifest. Output layout matches the original
yue2-music skill so abc_tools.py and the editing references keep working.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from abc_tools import parse_abc, report
from common import fresh_directory, read_json, sha256, write_json

DEFAULT_SERVER = os.environ.get("YUE2_SERVER", "http://127.0.0.1:8001")
TERMINAL = {"done", "failed", "cancelled"}


# ── HTTP ───────────────────────────────────────────────────────────────────────

class ServerError(RuntimeError):
    pass


class Server:
    def __init__(self, base, timeout=30):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _call(self, method, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("accept", "application/json")
        if data is not None:
            request.add_header("content-type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except ValueError:
                pass
            raise ServerError(f"{method} {path} -> {error.code}: {detail}") from None
        except urllib.error.URLError as error:
            raise ServerError(f"{method} {path}: cannot reach {self.base} ({error.reason})") from None

    def health(self):
        return self._call("GET", "/health")

    def submit(self, body):
        return self._call("POST", "/jobs", body)

    def status(self, job_id):
        return self._call("GET", f"/jobs/{job_id}")

    def wait(self, job_id, timeout):
        # Long-poll: returns as soon as the job is terminal, or the current view on timeout.
        return self._call("GET", f"/jobs/{job_id}/wait?timeout={int(timeout)}", timeout=timeout + 30)

    def cancel(self, job_id):
        try:
            return self._call("DELETE", f"/jobs/{job_id}")
        except ServerError as error:
            return {"error": str(error)}

    def download(self, job_id, name, destination):
        url = f"{self.base}/jobs/{job_id}/artifacts/{urllib.parse.quote(name)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response, \
                    Path(destination).open("wb") as stream:
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    stream.write(block)
        except urllib.error.HTTPError as error:
            raise ServerError(f"GET artifact {name} -> {error.code}") from None


def wait_for(server, job_id, poll_timeout, max_wait, quiet=False):
    started = time.monotonic()
    last_line = None
    while True:
        view = server.wait(job_id, poll_timeout)
        if view["status"] in TERMINAL:
            return view
        if not quiet:
            line = f"{job_id}: {view['status']} phase={view.get('phase')} tokens={view.get('tokens')} eta_s={view.get('eta_s')}"
            if line != last_line:
                print(line, file=sys.stderr, flush=True)
                last_line = line
        if max_wait and time.monotonic() - started > max_wait:
            server.cancel(job_id)
            raise TimeoutError(f"Job {job_id} exceeded --max-wait {max_wait}s and was cancelled")


# ── Artifacts ──────────────────────────────────────────────────────────────────

def fetch_artifacts(server, view, destination):
    """Download every artifact the server lists for a finished job, then verify hashes."""
    names = list(view.get("artifacts") or [])
    if not names:
        raise ServerError("Finished job lists no artifacts")
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ServerError(f"Refusing artifact path {name!r}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        server.download(view["id"], name, target)
    return names


def verify_manifest(destination, manifest_name, entries):
    """entries: {name: {"sha256": ..., "bytes": ...}} or {name: sha256} as YuE2 writes them."""
    for name, expected in entries.items():
        path = destination / name
        digest = expected["sha256"] if isinstance(expected, dict) else expected
        if not path.is_file():
            raise ServerError(f"{manifest_name} lists {name} but it was not downloaded")
        if sha256(path) != digest:
            raise ServerError(f"{name} does not match its {manifest_name} hash")
        if isinstance(expected, dict) and "bytes" in expected and path.stat().st_size != expected["bytes"]:
            raise ServerError(f"{name} size differs from {manifest_name}")


def verify_result(destination):
    """Standard-library equivalent of yue2.storage.verify_result for a downloaded generate job."""
    result = read_json(destination / "result.json")
    if result.get("status") != "complete":
        raise ServerError("Downloaded result did not complete")
    required = {"audio.flac", "prefix.npy", "semantic.npy", "latent.npy", "request.json", "config.json"}
    artifacts = result.get("artifacts", {})
    if not required <= set(artifacts):
        raise ServerError("Incomplete result artifact manifest")
    verify_manifest(destination, "result.json", artifacts)
    return result


def verify_plan(destination):
    manifest = read_json(destination / "plan_manifest.json")
    if not {"plan.json", "abc_tokens.npy", "prefix.npy"} <= set(manifest):
        raise ServerError("Incomplete saved plan")
    verify_manifest(destination, "plan_manifest.json", manifest)
    return read_json(destination / "plan.json")


# ── Requests ───────────────────────────────────────────────────────────────────

def request_data(args):
    data = read_json(args.request)
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    allowed = {"style", "tags", "lyrics", "cot", "seed", "abc", "abc_path", "cfg_scale", "id",
               "abc_sampling", "semantic_sampling"}
    if set(data) - allowed:
        raise ValueError(f"Unsupported request fields: {sorted(set(data) - allowed)}")
    if "tags" in data:
        tags = data.pop("tags")
        if "style" in data and data["style"] != tags:
            raise ValueError("style and tags disagree")
        data["style"] = tags
    sources = sum(x is not None for x in (data.get("abc"), data.get("abc_path"), args.abc_file))
    if sources > 1:
        raise ValueError("Use only one of abc, abc_path or --abc-file")
    abc_path = data.pop("abc_path", None)
    if abc_path is not None:
        data["abc"] = (Path(args.request).parent / abc_path).read_bytes().decode("utf-8")
    if args.abc_file:
        data["abc"] = Path(args.abc_file).read_bytes().decode("utf-8")
    if args.cot:
        data["cot"] = args.cot
    data.setdefault("cot", "full")
    data.setdefault("id", "song")
    if args.action == "all-modes" and data.get("abc") is not None:
        raise ValueError("all-modes requires text-only input, since off cannot accept ABC")
    if args.action == "plan" and data["cot"] == "off":
        raise ValueError("off has no symbolic plan; use generate")
    if data.get("abc") is not None:
        if data["cot"] == "off":
            raise ValueError("off cannot accept ABC")
        score = parse_abc(data["abc"])
        if data["cot"] == "melody" and any(v.chords for v in score.voices.values()):
            raise ValueError("melody input still contains chords; use abc_tools.py strip-chords")
    return data


def score_check(text, mode):
    if text is None:
        return {"status": "not_applicable"}
    try:
        score = parse_abc(text)
        if mode == "melody" and any(v.chords for v in score.voices.values()):
            raise ValueError("Melody-mode planner returned chord symbols")
        return {"status": "passed", "scope": "native ABC structure only", "score": report(score)}
    except ValueError as exc:
        return {"status": "failed", "error": str(exc)}


# ── Actions ────────────────────────────────────────────────────────────────────

def run_job(server, args, destination, body):
    """Submit one job, wait, download and verify. Returns the per-request result dict."""
    submitted = server.submit(body)
    job_id = submitted["id"]
    write_json(destination / "job.json", {"server": server.base, "id": job_id, "task": body["task"],
                                          "submitted": submitted})
    try:
        view = wait_for(server, job_id, args.wait_timeout, args.max_wait, quiet=args.quiet)
    except KeyboardInterrupt:
        print(f"Cancelling job {job_id}", file=sys.stderr)
        server.cancel(job_id)
        raise
    write_json(destination / "job.json", {"server": server.base, "id": job_id, "task": body["task"],
                                          "submitted": submitted, "final": view})
    if view["status"] != "done":
        raise ServerError(f"Job {job_id} {view['status']}: {view.get('error')}")
    fetch_artifacts(server, view, destination)
    return view


def generate_or_plan(server, args, output, data, health):
    if args.action == "all-modes":
        requests = [dict(data, cot=mode, id=f"{data['id']}_{mode}") for mode in ("full", "melody", "off")]
    else:
        requests = [dict(data)]
    task = "plan" if args.action == "plan" else "generate"
    write_json(output / "invocation.json", {
        "action": args.action, "server": server.base, "health": health,
        "task": task, "requests": requests,
    })
    results = []
    for request in requests:
        destination = fresh_directory(output / request["cot"]) if args.action == "all-modes" else output
        write_json(destination / "input.json", request)
        try:
            view = run_job(server, args, destination, dict(request, task=task))
            if task == "plan":
                plan = verify_plan(destination)
                abc = plan.get("abc")
                result = {"mode": request["cot"], "truncated": {"abc": plan.get("truncated", False)}}
            else:
                receipt = verify_result(destination)
                score = destination / "score.abc"
                abc = score.read_text(encoding="utf-8") if score.is_file() else None
                result = {"mode": request["cot"], "identity": receipt["identity"],
                          "truncated": receipt["truncated"], "audio_seconds": receipt["audio_seconds"]}
            check = score_check(abc, request["cot"])
            write_json(destination / "abc_check.json", check)
            result["job"] = view["id"]
            result["status"] = ("needs_review" if any(result["truncated"].values()) or check["status"] == "failed"
                                else "complete")
        except (ServerError, TimeoutError, OSError, ValueError) as exc:
            result = {"mode": request["cot"], "status": "failed", "error": str(exc), "type": type(exc).__name__}
            write_json(destination / "failure.json", result)
        results.append(result)
        print(json.dumps(result), flush=True)
    write_json(output / "run.json", {"results": results})
    return int(any(r["status"] != "complete" for r in results))


def decode(server, args, output, health):
    """Re-decode a finished generate job's cached latents with the chosen decoder (server task: decode)."""
    source = Path(args.source)
    original = verify_result(source)
    job = read_json(source / "job.json")
    if job.get("task") != "generate":
        raise ValueError("--source must be a directory produced by `run_yue2.py generate` against this server")
    write_json(output / "invocation.json", {"action": "decode", "server": server.base, "health": health,
                                            "source": str(source), "source_job": job["id"], "vae": args.vae})
    write_json(output / "source_generation.json", {
        "source_result_sha256": sha256(source / "result.json"),
        "source_latent_sha256": sha256(source / "latent.npy"),
        "source_semantic_sha256": sha256(source / "semantic.npy"),
        "config": read_json(source / "config.json"), "identity": original["identity"],
        "weights": original["weights"],
    })
    view = run_job(server, args, output, {"task": "decode", "source_job": job["id"], "vae": args.vae})
    receipt = verify_result(output)
    if sha256(output / "latent.npy") != sha256(source / "latent.npy"):
        raise ServerError("Latents changed during re-decoding")
    result = {"status": "complete", "job": view["id"], "identity": receipt["identity"],
              "truncated": receipt["truncated"], "audio_seconds": receipt["audio_seconds"]}
    write_json(output / "run.json", result)
    print(json.dumps(result), flush=True)
    return int(any(result["truncated"].values()))


def run(args):
    data = None if args.action == "decode" else request_data(args)
    server = Server(args.server, timeout=args.http_timeout)
    health = server.health()
    output = fresh_directory(args.output)
    try:
        if args.action == "decode":
            return decode(server, args, output, health)
        return generate_or_plan(server, args, output, data, health)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error": str(exc), "type": type(exc).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("generate", "all-modes", "plan", "decode"):
        command = sub.add_parser(action)
        command.add_argument("--output", required=True, type=Path, help="Fresh destination; no automatic overwrite/resume")
        if action == "decode":
            command.add_argument("--source", required=True, type=Path, help="Directory produced by `generate`")
            command.add_argument("--vae", choices=("standard", "legacy"), default="legacy",
                                 help="Decoder: standard (listening) or legacy (benchmark protocol)")
        else:
            command.add_argument("--request", required=True, type=Path)
            command.add_argument("--cot", choices=("full", "melody", "off"))
            command.add_argument("--abc-file", type=Path)
        command.add_argument("--server", default=DEFAULT_SERVER, help=f"yue2-server base URL (env YUE2_SERVER; default {DEFAULT_SERVER})")
        command.add_argument("--wait-timeout", type=int, default=300, help="Seconds per long-poll request")
        command.add_argument("--max-wait", type=int, default=0, help="Give up and cancel after this many seconds (0 = never)")
        command.add_argument("--http-timeout", type=int, default=60, help="Timeout for ordinary HTTP calls and downloads")
        command.add_argument("--quiet", action="store_true", help="No progress lines on stderr")
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
