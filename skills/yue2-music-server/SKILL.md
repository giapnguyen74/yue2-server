---
name: yue2-music-server
description: Generate, cover, transcribe, and edit songs with YuE2 through a running yue2-server (HTTP job queue; no local model loading) and SheetSage2/MERT2. Use for YuE2 full/melody/off generation, audio-to-ABC covers, style or lyric changes, score editing, agentic reharmonization, melody preservation, singable lyric adaptation, and reproducible before/after comparisons; also for YuE2 生成、翻唱、改编、改谱、换词和智能体编辑.
---

# YuE2 Music (server)

Turn a musical request into a reproducible song and an audible comparison. Use released
model interfaces. Retain an original song and its plan before making changes.

This is the `yue2-music` skill adapted to a running **yue2-server**: `scripts/run_yue2.py`
and `scripts/transcribe.py` submit jobs to it, wait, and download the native artifacts
instead of loading models themselves. Every helper is standard-library Python 3.10+; the
client machine needs no GPU, no torch and no model files.

## Choose the workflow

| Request | Workflow |
| --- | --- |
| Generate with editable melody and harmony | YuE2 `cot="full"` → ABC → song |
| Generate with a melody plan and free accompaniment | YuE2 `cot="melody"` → chord-free ABC → song |
| Generate without symbolic planning | YuE2 `cot="off"` → song; no editable ABC |
| Cover a recording | SheetSage2 → inspect/correct ABC → strip chords → YuE2 `melody` |
| Cover an ABC melody | Inspect/convert native ABC → strip chords → YuE2 `melody` |
| Change harmony, instruments, tempo, structure, or lyrics | Copy full plan → edit ABC/text → regenerate |
| Agentic editing | Export plan/baseline → bounded editing agent → check invariants → render → compare |
| Check sung lyrics, lyric rewrite or translation | `lyrics.py` → Qwen3-ASR transcript → WER/CER and PER against the intended lyrics |
| Analyze musical features | Use MERT2 only when continuous features are needed |

```text
audio → SheetSage2 [loads MERT-v2-FullSong itself] → ABC
style + lyrics → YuE2 full/melody planning        → ABC
                                                  edit/validate
style + lyrics + ABC → YuE2 semantic generation → synthesis → latents → VAE → song
style + lyrics      → YuE2 off generation      → synthesis → latents → VAE → song
```

Do not feed public MERT feature tensors to YuE2 as codec tokens. YuE2 exposes no
audio-reference, phoneme-alignment, or local-inpainting argument.

## Use the server through the helpers

The server owns the GPU and the model snapshots (YuE2 and SheetSage2) and runs one job at
a time. The helpers find it at `$YUE2_SERVER` or `--server` (default
`http://127.0.0.1:8001`), check that it is reachable, and record its reported model and
decoder identities in each run's `invocation.json`. Model selection, revisions, VRAM
budget and offline mode are the server operator's settings: there are no `--model`,
`--vae`, `--revision` or `--offline` options here.

Every helper waits for its job and prints the phase and token count on stderr. Songs take
minutes; `--max-wait N` cancels a job that runs past N seconds, and Ctrl-C cancels the
running job before exiting. Use fresh output directories; each records the server job id
in `job.json`. Only `transcribe.py` uploads a file; `--upload-timeout` covers large
recordings.

This skill's original instructions, helpers, and templates are licensed under
[Apache 2.0](LICENSE). Copyright (c) 2026 the YuE2 authors. Model weights and third-party
dependencies retain their applicable licenses.

Do not silently shorten a requested song or lower inference settings to hide a failure;
report the server's error. Use `YuE2-Vae` for listening and `YuE2-Vae-legacy` when
reproducing the supplied benchmark protocol (`run_yue2.py decode --vae legacy` once the
server has that decoder). Keep decoded files separate. Do not infer their roles from the
word “legacy.”

Run helper paths below relative to this skill folder.

## Generate and retain the plan

Start with [assets/prompt.json](assets/prompt.json), an original example. Put genre,
instruments, vocal character, language and intended tempo in `style`; put section tags
and actual words in `lyrics`. Keep implementation notes out of lyrics.

```bash
python scripts/run_yue2.py generate --request assets/prompt.json --output outputs/pop
python scripts/run_yue2.py all-modes --request assets/prompt.json --output outputs/modes
python scripts/run_yue2.py plan --request assets/prompt.json --output outputs/plan
```

Each call submits a job, prints its phase and token count on stderr while waiting, and
downloads the native artifacts. Inspect `result.json`, truncation, `score.abc`,
`request.json`, and audio. Keep exact tokens and `latent.npy`; the helper verifies every
file against the server's hashes. Preserve all requested modes and failures. A successful
process or playable file does not establish musical quality.

Read [generation-and-covers.md](references/generation-and-covers.md) for the request
fields, CFG, sampling and mode semantics; its Python examples describe what the server
does internally. Submit modified ABC as a new input. Long songs take minutes; `--max-wait`
cancels a job that runs past a limit, otherwise the helper waits for completion.

## Cover a recording

1. Transcribe through the server. Select the vocal melody or the full lead melody,
   including instrumental passages.
2. Inspect warnings and correct missed notes, meter or key before attributing errors
   to YuE2. Preserve source audio and raw transcription.
3. Export chord-free ABC. Select a retained voice explicitly when dropping a part;
   removing chords alone should preserve both melodic voices and their rests.
4. Render with `cot="melody"`, target style and suitable lyrics. This supplies a symbolic
   melody condition; it does not preserve the source singer's identity or waveform.

```bash
# Uploads the recording; SheetSage2 runs on the server.
python scripts/transcribe.py reference.wav --task melody-full --output outputs/transcription
python scripts/abc_tools.py strip-chords outputs/transcription/score.abc outputs/cover.abc

# Any Python; the request supplies target style and lyrics, the server renders.
python scripts/run_yue2.py generate --request assets/prompt.json --cot melody \
  --abc-file outputs/cover.abc --output outputs/cover-song
```

`cot="melody"` does not remove chord symbols automatically. To retain the original
harmony as well, use full transcription and `cot="full"`; call this score-conditioned
regeneration with melody and harmony.

## Edit or delegate an edit

Read [editing-workflows.md](references/editing-workflows.md) and
[abc-editing.md](references/abc-editing.md) before changing a score.

1. Render a baseline from the full plan. Freeze its original directory.
2. Define invariants: exact pitches; pitch plus rhythm; contour only; or bounded melodic
   adaptation. Specify voices/passages, lyrics, instruments, tempo, meter and structure.
3. If delegation is available, give a score-editing agent raw ABC, prompt, lyrics, the
   requested change and the [edit brief](assets/edit-brief.md). Request a new ABC,
   revised style/lyrics as needed, and an edit manifest. Give a separate reviewer the
   before/after artifacts and constraints. Without delegation, perform these stages
   yourself. Keep model generation sequential per GPU.
4. Check musical events, not character strings: ties, accidentals and compressed rests
   matter. Run:

   ```bash
   python scripts/abc_tools.py inspect edits/jazz.abc
   python scripts/abc_tools.py compare outputs/plan/score.abc edits/jazz.abc --voices Vocal
   python scripts/run_yue2.py generate --request edits/jazz.json --cot full \
     --abc-file edits/jazz.abc --output outputs/jazz
   ```

   Add `--allow-tempo-change` for intentional tempo changes. Exact comparison should
   fail for intentional rhythm changes; audit permitted differences from its report
   instead of relabeling the result “melody preserved.”
   Keep the edited score connected through `--abc-file` or request `abc_path`;
   omitting both with `abc: null` generates a fresh plan and discards the edit.
5. Regenerate after changing style, lyrics or ABC. Old acoustic latents can be decoded
   again, but cannot implement a musical or lyric edit.
6. Compare full songs and short passages around the edit. Revise when the requested
   effect fails; retain each attempt and its actual prompt.

For lyric translation, adapt syllables, stress, vowels and breath points. Keep a
syllable/phoneme-to-note sidecar. Do not invent a `phonemes` field or mistake the sidecar
for hard acoustic alignment. Use ASR/PER and listening as separate evidence.

## Check the sung lyrics

The server runs Qwen3-ASR (the model behind WildSongBench's PER metric; open weights,
best published results on sung-lyrics benchmarks) and scores the transcript against the
lyrics that were requested:

```bash
python scripts/lyrics.py --source outputs/jazz --output outputs/jazz-lyrics
python scripts/lyrics.py --source outputs/translated --lyrics-file translated.txt \
  --language Chinese --passes 4 --output outputs/translated-lyrics
```

`--source` reuses the server's copy of the audio and takes the reference from the run's
`request.json`; `--lyrics-file` overrides it, and a plain audio path uploads instead.
Read `lyrics_asr.json`: `per` (phoneme error rate, lower is better; the YuE2 benchmark's
best systems sit around 6–10 %), `unit_error_rate` (WER for English, CER for Chinese),
the detected language and every pass's transcript. `--passes N` keeps the lowest-PER pass,
the shape of the benchmark protocol; the scoring itself is the documented one in the
report's `scoring` field, not the benchmark's undisclosed evaluator, so compare versions
against each other rather than against published tables. Inspect the transcript for the
dropped or repeated words a scalar hides; a low PER on a translation says the new words are
intelligible, not that they are good lyrics.

## Deliver an audible result

Return the playable audio (`audio.mp3` for delivery, `audio.flac` lossless), the full
prompt and lyrics, before/after ABC, the invariant check reports and any requested
evaluations, each from its own output directory. Keep model/decoder identity (from
`invocation.json` and `result.json`), seeds, truncation flags and failures visible.
Distinguish symbolic checks, ASR, listening and quality scores; an agent that cannot
play audio says which checks it actually performed. Do not claim exact note realization,
instrument removal or sample-accurate preservation from an ABC check or SongBench score
alone.
