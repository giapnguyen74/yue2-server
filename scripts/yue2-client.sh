#!/usr/bin/env bash
# Minimal curl/jq client for yue2-server, for shell-driven agents. The yue2-music-server skill's
# Python helpers do more (hash checks, native layout); this is the same contract in 60 lines.
#
#   yue2-client.sh submit request.json          # -> job id on stdout
#   yue2-client.sh wait <id>                    # long-polls; exit 0 done, 1 failed/cancelled, 2 truncated
#   yue2-client.sh fetch <id> <dir>             # audio.mp3, audio.flac, score.abc, result.json into <dir>
#   yue2-client.sh song request.json <dir>      # submit + wait + fetch
#   yue2-client.sh transcribe song.wav <dir> [melody-full|melody-vocal|full]
#   yue2-client.sh cancel <id> | status <id> | health
#
# YUE2_SERVER selects the server (default http://127.0.0.1:8001).
set -euo pipefail
S=${YUE2_SERVER:-http://127.0.0.1:8001}
for tool in curl jq; do command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 2; }; done

api() { curl -sS --fail-with-body "$@"; }

submit()  { api -X POST "$S/jobs" -H 'content-type: application/json' --data-binary "@$1" | jq -r .id; }
status()  { api "$S/jobs/$1"; }
health()  { api "$S/health"; }
cancel()  { api -X DELETE "$S/jobs/$1"; }

wait_job() {
    local id=$1 view st
    while :; do
        view=$(api "$S/jobs/$id/wait?timeout=300")
        st=$(jq -r .status <<<"$view")
        case "$st" in
            done) jq -c '{id,status,audio_seconds,truncated,identity}' <<<"$view"
                  [[ $(jq '[.truncated // {} | .[]] | any' <<<"$view") == true ]] && return 2 || return 0 ;;
            failed|cancelled) jq -c '{id,status,error}' <<<"$view" >&2; return 1 ;;
            *) jq -r '"\(.id): \(.status) phase=\(.phase) tokens=\(.tokens) eta_s=\(.eta_s)"' <<<"$view" >&2 ;;
        esac
    done
}

fetch() {
    local id=$1 dir=$2 name
    mkdir -p "$dir"
    for name in $(api "$S/jobs/$id" | jq -r '.artifacts[]? | select(. == "audio.mp3" or . == "audio.flac" or . == "score.abc" or . == "result.json" or . == "transcription_manifest.json")'); do
        api -o "$dir/$name" "$S/jobs/$id/artifacts/$name"
    done
    ls "$dir"
}

transcribe() {
    local file=$1 dir=$2 task=${3:-melody-full} id
    id=$(api -X POST "$S/jobs/transcribe" -F "audio=@$file" -F "task=$task" | jq -r .id)
    wait_job "$id" || true
    fetch "$id" "$dir"
}

cmd=${1:-}; shift || true
case "$cmd" in
    submit) submit "$1" ;;
    wait) wait_job "$1" ;;
    fetch) fetch "$1" "$2" ;;
    song) id=$(submit "$1"); echo "job $id" >&2; rc=0; wait_job "$id" || rc=$?; [[ $rc -ne 1 ]] && fetch "$id" "$2"; exit $rc ;;
    transcribe) transcribe "$@" ;;
    cancel) cancel "$1" ;;
    status) status "$1" ;;
    health) health ;;
    *) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
