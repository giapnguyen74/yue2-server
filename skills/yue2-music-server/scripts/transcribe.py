#!/usr/bin/env python3
"""Transcribe audio to native ABC through a running yue2-server (SheetSage2 runs server-side).

Standard library only; the client needs no GPU, torch or model files. The audio file is
uploaded as multipart form data to POST /jobs/transcribe, the job is awaited like any
other, and the server-side worker's artifacts (score.abc, MIDI, LAB annotations,
transcription_manifest.json, ...) are downloaded and hash-checked. Output layout matches
the original skill, so abc_tools.py strip-chords and the cover workflow are unchanged.
"""

import argparse
import json
import mimetypes
import sys
import urllib.request
import uuid
from pathlib import Path

from abc_tools import parse_abc, report
from common import fresh_directory, read_json, sha256, write_json
from run_yue2 import DEFAULT_SERVER, Server, ServerError, fetch_artifacts, verify_manifest, wait_for


def multipart(fields, file_field, path):
    """Encode text fields plus one file as multipart/form-data (RFC 7578)."""
    boundary = "----yue2-" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        if value is None:
            continue
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode("utf-8")
    kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
             f"filename=\"{path.name}\"\r\nContent-Type: {kind}\r\n\r\n").encode("utf-8")
    body += path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def submit_transcription(server, audio, fields, timeout):
    data, content_type = multipart(fields, "audio", audio)
    request = urllib.request.Request(server.base + "/jobs/transcribe", data=data, method="POST")
    request.add_header("content-type", content_type)
    request.add_header("accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        raise ServerError(f"POST /jobs/transcribe -> {error.code}: {detail}") from None
    except urllib.error.URLError as error:
        raise ServerError(f"POST /jobs/transcribe: cannot reach {server.base} ({error.reason})") from None


def run(args):
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive and explicitly crops the input")
    server = Server(args.server, timeout=args.http_timeout)
    health = server.health()
    output = fresh_directory(args.output)
    fields = {"task": args.task, "preset": args.preset, "max_seconds": args.max_seconds, "dtype": args.dtype}
    write_json(output / "input.json", {
        "source_name": args.audio.name, "source_audio_sha256": sha256(args.audio),
        "server": server.base, "health": health, "options": fields,
    })
    try:
        submitted = submit_transcription(server, args.audio, fields, timeout=args.upload_timeout)
        job_id = submitted["id"]
        write_json(output / "job.json", {"server": server.base, "id": job_id, "task": "transcribe", "submitted": submitted})
        try:
            view = wait_for(server, job_id, args.wait_timeout, args.max_wait, quiet=args.quiet)
        except KeyboardInterrupt:
            print(f"Cancelling job {job_id}", file=sys.stderr)
            server.cancel(job_id)
            raise
        write_json(output / "job.json", {"server": server.base, "id": job_id, "task": "transcribe",
                                         "submitted": submitted, "final": view})
        if view["status"] != "done":
            raise ServerError(f"Job {job_id} {view['status']}: {view.get('error')}")
        fetch_artifacts(server, view, output)
        manifest = read_json(output / "transcription_manifest.json")
        if manifest.get("status") != "complete":
            raise ServerError("Transcription manifest is not complete")
        if manifest.get("source_audio_sha256") != sha256(args.audio):
            raise ServerError("Server transcribed a different audio file than the one uploaded")
        verify_manifest(output, "transcription_manifest.json",
                        {name: digest for name, digest in manifest["artifacts"].items()
                         if name != "transcription_manifest.json"})
        abc = (output / "score.abc").read_text(encoding="utf-8")
        score = parse_abc(abc)
        if args.task != "full" and any(v.chords for v in score.voices.values()):
            raise ValueError("Melody transcription contains unexpected chord symbols")
        write_json(output / "abc_check.json", {"status": "passed", "score": report(score),
                   "scope": "symbolic format; transcription accuracy still needs review"})
        print(f"Saved {output / 'score.abc'}; warnings: {manifest.get('warnings', [])}")
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--task", choices=("full", "melody-full", "melody-vocal"), default="full")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--preset", choices=("default", "paper"), default="default")
    parser.add_argument("--max-seconds", type=float, help="Explicitly crop audio; omitted means process the whole input")
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"yue2-server base URL (env YUE2_SERVER; default {DEFAULT_SERVER})")
    parser.add_argument("--upload-timeout", type=int, default=600, help="Seconds allowed for the audio upload")
    parser.add_argument("--wait-timeout", type=int, default=300, help="Seconds per long-poll request")
    parser.add_argument("--max-wait", type=int, default=0, help="Give up and cancel after this many seconds (0 = never)")
    parser.add_argument("--http-timeout", type=int, default=60)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
