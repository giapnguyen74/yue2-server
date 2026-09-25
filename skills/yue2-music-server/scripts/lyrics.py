#!/usr/bin/env python3
"""Recognise the sung lyrics of a song with the server's Qwen3-ASR and score them against reference lyrics.

  lyrics.py --source outputs/pop --output outputs/pop-lyrics                 # a run_yue2.py output directory
  lyrics.py song.mp3 --lyrics-file translated.txt --output outputs/check     # any audio file, uploaded
  options: --language auto|English|Chinese, --passes N (best of N), --server, --wait-timeout, --max-wait

Use it after a lyric rewrite or translation: the reference is the text you asked YuE2 to sing, the score
says how much of it is intelligible. Outputs mirror the worker: transcript.txt, lyrics_asr.json (every
pass, unit error rate = WER for English or CER for Chinese, PER over phonemes) and lyrics_manifest.json,
hash-checked. Without --lyrics-file only a transcript is produced. Standard library only.
"""

import argparse
import json
import sys
from pathlib import Path

from common import fresh_directory, read_json, sha256, write_json
from run_yue2 import DEFAULT_SERVER, Server, ServerError, fetch_artifacts, verify_manifest, wait_for
from transcribe import multipart, submit_upload


def run(args):
    if (args.source is None) == (args.audio is None):
        raise ValueError("Give exactly one of --source <run_yue2.py output dir> or an audio file")
    server = Server(args.server, timeout=args.http_timeout)
    health = server.health()
    if not health.get("lyrics", False):
        raise ServerError("the server has lyrics recognition disabled (Qwen3-ASR not installed)")
    reference = args.lyrics_file.read_text(encoding="utf-8") if args.lyrics_file else None
    if reference is None and args.source is not None:
        request = args.source / "request.json"
        if request.is_file():
            reference = read_json(request).get("lyrics")
    output = fresh_directory(args.output)
    fields = {"lyrics": reference, "language": args.language, "passes": args.passes}
    write_json(output / "input.json", {"source": str(args.source) if args.source else None,
                                       "audio": str(args.audio) if args.audio else None,
                                       "server": server.base, "health": health, "options": fields})
    try:
        if args.source is not None:
            job = read_json(args.source / "job.json")
            submitted = server.submit({"task": "lyrics", "source_job": job["id"], **fields})
        else:
            if not args.audio.is_file():
                raise FileNotFoundError(args.audio)
            submitted = submit_upload(server, "/jobs/lyrics", args.audio, fields, timeout=args.upload_timeout)
        job_id = submitted["id"]
        write_json(output / "job.json", {"server": server.base, "id": job_id, "task": "lyrics", "submitted": submitted})
        try:
            view = wait_for(server, job_id, args.wait_timeout, args.max_wait, quiet=args.quiet)
        except KeyboardInterrupt:
            server.cancel(job_id)
            raise
        write_json(output / "job.json", {"server": server.base, "id": job_id, "task": "lyrics",
                                         "submitted": submitted, "final": view})
        if view["status"] != "done":
            raise ServerError(f"Job {job_id} {view['status']}: {view.get('error')}")
        fetch_artifacts(server, view, output)
        manifest = read_json(output / "lyrics_manifest.json")
        verify_manifest(output, "lyrics_manifest.json",
                        {k: v for k, v in manifest["artifacts"].items() if k != "lyrics_manifest.json"})
        if args.audio is not None and manifest.get("source_audio_sha256") != sha256(args.audio):
            raise ServerError("Server scored a different audio file than the one uploaded")
        report = read_json(output / "lyrics_asr.json")
        summary = {"status": "complete", "job": job_id, "per": report.get("per"),
                   "unit_error_rate": report.get("unit_error_rate"), "language": report.get("language_detected"),
                   "best_pass": report.get("best_pass"), "transcript": str(output / "transcript.txt")}
        write_json(output / "run.json", summary)
        print(json.dumps(summary, ensure_ascii=False))
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="?", type=Path, help="Audio file to upload (or use --source)")
    parser.add_argument("--source", type=Path, help="run_yue2.py output directory; its request.json supplies the reference")
    parser.add_argument("--lyrics-file", type=Path, help="Reference lyrics; overrides the source's request.json")
    parser.add_argument("--language", default="auto", help="auto, English or Chinese")
    parser.add_argument("--passes", type=int, default=1, help="ASR passes; the lowest-PER pass is reported")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--upload-timeout", type=int, default=600)
    parser.add_argument("--wait-timeout", type=int, default=300)
    parser.add_argument("--max-wait", type=int, default=0)
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
