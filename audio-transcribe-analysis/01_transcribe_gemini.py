#!/usr/bin/env python3
"""
Stage 1: transcribes caregiver/navigator call recordings via Gemini's native audio understanding
(Vertex AI) -- no separate ASR model, no local diarization library. Gemini takes the raw audio
directly and returns a turn-by-turn transcript with its own best-effort speaker labels.

This replaces the local faster-whisper + pyannote.audio stack in ../call-signal-pipeline/ with a
single Gemini call per file. Trade-off, stated plainly: faster-whisper + pyannote gives you a
purpose-built, locally-verifiable diarization model; asking Gemini to transcribe AND separate
speakers in one pass means the speaker split is whatever the model infers from voice/content cues,
not a dedicated diarization algorithm. Spot-check `diarization_confidence` and a handful of
transcripts against the audio before trusting the speaker split at scale -- same caveat this
project has applied to every other self-reported confidence signal so far.

Why Gemini can do this in one pass: the calls are mono (single channel, confirmed via ffprobe --
see ../call-signal-pipeline/README.md), 8kHz telephony-quality audio, so there is no free
stereo-channel trick to separate speakers, and a multimodal model reading the actual audio (tone,
turn-taking, content) has more to work with than a transcript-only pass would.

Input size handling:
  - Files under --inline-limit-mb (default 15MB -- comfortably under Vertex AI's request-size
    ceiling for inline bytes) are sent directly as base64-encoded bytes, same pattern as the PDF
    extraction scripts in ../../home-assessment-forms/.
  - Larger files require --gcs-bucket: the script uploads the audio to
    gs://<bucket>/call-sentiment-pipeline/<filename> first and references it by URI instead of
    inlining it. (A ~750KB, ~3-minute sample call is nowhere near this threshold -- this path is
    for longer recordings you haven't hit yet.)

Usage:
    python 01_transcribe_gemini.py \\
        --input-dir /path/to/Sample \\
        --project-id avvacare-clinical-ops \\
        --location us-central1 \\
        --out-prefix ./transcripts \\
        [--model gemini-2.5-flash] [--workers 3] [--limit 5] [--gcs-bucket my-bucket]

Outputs (checkpointed, one file per call under --work-dir so a killed/interrupted run resumes
instead of re-transcribing everything):
    {work-dir}/<call_id>/transcript.json  -- one record per call
    {out-prefix}.json                     -- all records, same shape as extract_*_gemini.py outputs
    {out-prefix}.csv                      -- flattened, one row per call, for a quick skim
"""
import argparse
import csv
import json
import mimetypes
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    from google import genai
    from google.genai import types
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests
except ImportError:
    sys.exit("Missing dependency: pip install google-genai google-api-core")

RATE_LIMIT_ERRORS = (ResourceExhausted, TooManyRequests)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
MIME_OVERRIDES = {".mp3": "audio/mp3", ".m4a": "audio/mp4"}

INLINE_LIMIT_MB_DEFAULT = 15

TRANSCRIPT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "turns": {
            "type": "ARRAY",
            "description": "Verbatim, turn-by-turn transcript in chronological order.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "speaker": {
                        "type": "STRING",
                        "description": "Your own best-effort speaker label based on voice/content "
                                       "cues -- 'Speaker 1', 'Speaker 2', etc., consistent across "
                                       "the whole call. Not a verified diarization.",
                    },
                    "start_time_sec": {
                        "type": "NUMBER", "nullable": True,
                        "description": "Approximate turn start time in seconds from call start, "
                                       "if you can estimate it; null if not confident.",
                    },
                    "text": {"type": "STRING", "description": "Verbatim words spoken this turn, in the "
                                                               "language actually spoken."},
                    "text_en": {
                        "type": "STRING", "nullable": True,
                        "description": "English translation of this turn's text. Required whenever "
                                       "the call is not in English -- this is what lets a reviewer "
                                       "who doesn't speak the call's language validate the transcript "
                                       "and downstream sentiment scoring. Null only when the call is "
                                       "already in English (no translation needed).",
                    },
                },
                "required": ["speaker", "text"],
            },
        },
        "num_speakers_detected": {"type": "INTEGER"},
        "likely_navigator_speaker": {
            "type": "STRING", "nullable": True,
            "description": "Which speaker label is most likely the AvvaCare caregiver/navigator "
                           "(vs. the patient/family member), based on who is asking questions, "
                           "referencing care plans, or leading the call. Null if unclear.",
        },
        "language": {"type": "STRING", "description": "Primary language spoken, e.g. 'English'."},
        "audio_quality_issues": {
            "type": "ARRAY", "items": {"type": "STRING"},
            "description": "Notable quality problems that hurt transcription confidence -- "
                           "crosstalk, long silences, heavy accent, garbled audio, hold music, "
                           "voicemail, etc. Empty list if the audio was clean.",
        },
        "diarization_confidence": {
            "type": "STRING", "enum": ["high", "medium", "low"],
            "description": "Your own confidence in the speaker split specifically, separate from "
                           "transcription accuracy -- 'low' if speakers were hard to tell apart "
                           "(similar voices, heavy crosstalk, single very short call).",
        },
    },
    "required": ["turns", "num_speakers_detected", "diarization_confidence"],
}

PROMPT = """You are transcribing a recorded phone call between an AvvaCare caregiver navigator and
a patient or family member. The audio is mono (both speakers on one channel), telephony-quality
(8kHz), and may include hold music, voicemail, or crosstalk.

Produce a verbatim, turn-by-turn transcript. Label speakers consistently as "Speaker 1", "Speaker
2", etc., based on voice and content cues -- switch labels only when the speaker actually changes,
not on every breath or pause. Include filler words and false starts as actually spoken; do not
clean up or paraphrase.

Make your best guess at which speaker is the AvvaCare navigator (the person asking check-in
questions, referencing the patient's care plan, or leading the call structure) versus the patient
or family member, but say so honestly if it's genuinely unclear.

If the call is not in English, fill in `text_en` for every turn with a natural, accurate English
translation (not a word-for-word gloss) alongside the original-language `text`. This is what lets
someone who doesn't speak the call's language check the transcript and the sentiment analysis that
follows against what was actually said. Leave `text_en` null only when the call is already in
English.

Separately from how well you could hear the words, rate your own confidence in the SPEAKER SPLIT
specifically (diarization_confidence): mark it "low" whenever voices were hard to tell apart, there
was heavy crosstalk, or the call was too short/one-sided to be sure. This is used to decide which
calls get a human's ears before anything downstream trusts the speaker split."""


def audio_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in MIME_OVERRIDES:
        return MIME_OVERRIDES[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "audio/mpeg"


def upload_to_gcs(local_path: Path, bucket: str) -> str:
    try:
        from google.cloud import storage
    except ImportError:
        sys.exit("Missing dependency for --gcs-bucket: pip install google-cloud-storage")
    client = storage.Client()
    blob_name = f"call-sentiment-pipeline/{local_path.name}"
    blob = client.bucket(bucket).blob(blob_name)
    if not blob.exists():
        blob.upload_from_filename(str(local_path))
    return f"gs://{bucket}/{blob_name}"


def build_audio_part(audio_path: Path, inline_limit_mb: float, gcs_bucket: Optional[str]) -> "types.Part":
    mime = audio_mime_type(audio_path)
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb <= inline_limit_mb:
        return types.Part.from_bytes(data=audio_path.read_bytes(), mime_type=mime)
    if not gcs_bucket:
        raise ValueError(
            f"{audio_path.name} is {size_mb:.1f}MB, over --inline-limit-mb ({inline_limit_mb}MB) "
            f"-- pass --gcs-bucket to upload large files instead of inlining them."
        )
    gcs_uri = upload_to_gcs(audio_path, gcs_bucket)
    return types.Part.from_uri(file_uri=gcs_uri, mime_type=mime)


def transcribe_one_call(client, model: str, audio_path: Path, inline_limit_mb: float,
                         gcs_bucket: Optional[str]) -> dict:
    audio_part = build_audio_part(audio_path, inline_limit_mb, gcs_bucket)
    response = client.models.generate_content(
        model=model,
        contents=[audio_part, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TRANSCRIPT_SCHEMA,
        ),
    )
    parsed = json.loads(response.text)
    parsed["source_file"] = audio_path.name
    return parsed


def transcribe_with_retry(client, model, audio_path, inline_limit_mb, gcs_bucket, max_attempts=5) -> dict:
    for attempt in range(1, max_attempts + 1):
        try:
            return transcribe_one_call(client, model, audio_path, inline_limit_mb, gcs_bucket)
        except RATE_LIMIT_ERRORS:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)  # 2, 4, 8, 16s


def full_text(record: dict) -> str:
    return "\n".join(f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in record.get("turns", []))


def full_text_en(record: dict) -> str:
    """English-readable transcript: uses text_en where present (non-English calls), falls back to
    text otherwise -- this is what a reviewer who doesn't speak the call's language should read,
    not full_text()."""
    return "\n".join(
        f"{t.get('speaker', '?')}: {t.get('text_en') or t.get('text', '')}" for t in record.get("turns", [])
    )


def flatten_for_csv(record: dict) -> dict:
    is_translated = record.get("language", "").strip().lower() not in ("", "english")
    return {
        "source_file": record.get("source_file"),
        "num_speakers_detected": record.get("num_speakers_detected"),
        "likely_navigator_speaker": record.get("likely_navigator_speaker"),
        "diarization_confidence": record.get("diarization_confidence"),
        "language": record.get("language"),
        "audio_quality_issues": "; ".join(record.get("audio_quality_issues", []) or []),
        "num_turns": len(record.get("turns", [])),
        # For a non-English call this is the English translation, not the original -- flagged via
        # transcript_preview_is_translated so nobody mistakes it for the verbatim transcript.
        "transcript_preview": (full_text_en(record) if is_translated else full_text(record))[:500],
        "transcript_preview_is_translated": is_translated,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--model", default="gemini-2.5-flash",
                     help="gemini-2.5-flash by default. Try gemini-2.5-pro if diarization_confidence "
                          "comes back 'low' on a lot of calls -- the larger model tends to hold "
                          "speaker identity across turns more reliably on noisy telephony audio.")
    ap.add_argument("--work-dir", default="work",
                     help="Per-call checkpoint directory -- a call already transcribed here is "
                          "skipped on re-run, so an interrupted batch resumes instead of restarting.")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--workers", type=int, default=3,
                     help="Concurrent calls in flight. Kept lower than the home-assessment "
                          "pipeline's default (5): audio inputs are larger and slower for Gemini "
                          "to process than a single-page form scan, so fewer, larger requests in "
                          "flight is the safer default against Vertex AI's per-project rate limits.")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N files.")
    ap.add_argument("--inline-limit-mb", type=float, default=INLINE_LIMIT_MB_DEFAULT,
                     help="Files at or under this size are sent as inline bytes. Larger files "
                          "require --gcs-bucket.")
    ap.add_argument("--gcs-bucket", default=None,
                     help="GCS bucket to stage audio files over --inline-limit-mb. Not needed for "
                          "the current ~750KB sample calls; required once you process longer "
                          "recordings.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    audio_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
    if args.limit:
        audio_paths = audio_paths[: args.limit]
    if not audio_paths:
        sys.exit(f"No audio files ({', '.join(sorted(AUDIO_EXTENSIONS))}) found in {input_dir}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(vertexai=True, project=args.project_id, location=args.location)

    def call_output_path(audio_path: Path) -> Path:
        call_dir = work_dir / audio_path.stem
        call_dir.mkdir(parents=True, exist_ok=True)
        return call_dir / "transcript.json"

    to_process = []
    records = []
    for p in audio_paths:
        out_path = call_output_path(p)
        if out_path.exists():
            records.append(json.loads(out_path.read_text()))
        else:
            to_process.append(p)

    print(f"Found {len(audio_paths)} audio file(s): {len(records)} already transcribed (checkpoint "
          f"hit), {len(to_process)} to process now ({args.workers} concurrent).")

    def process_one(audio_path: Path):
        record = transcribe_with_retry(client, args.model, audio_path, args.inline_limit_mb, args.gcs_bucket)
        call_output_path(audio_path).write_text(json.dumps(record, indent=2))
        return audio_path, record

    failed = []
    if to_process:
        done_count = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, p): p for p in to_process}
            for future in as_completed(futures):
                audio_path = futures[future]
                done_count += 1
                try:
                    _, record = future.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[{done_count}/{len(to_process)}] {audio_path.name} ... FAILED ({e})")
                    failed.append((audio_path.name, str(e)))
                    continue
                flag = " -- LOW DIARIZATION CONFIDENCE" if record.get("diarization_confidence") == "low" else ""
                print(f"[{done_count}/{len(to_process)}] {audio_path.name} ... ok "
                      f"({record.get('num_speakers_detected')} speaker(s)){flag}")
                records.append(record)

    order = {p.name: i for i, p in enumerate(audio_paths)}
    records.sort(key=lambda r: order.get(r["source_file"], 999999))

    out_json_path = Path(f"{args.out_prefix}.json")
    out_csv_path = Path(f"{args.out_prefix}.csv")
    out_json_path.write_text(json.dumps(records, indent=2))

    flat_records = [flatten_for_csv(r) for r in records]
    fieldnames = list(flat_records[0].keys()) if flat_records else []
    with out_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_records)

    if failed:
        print(f"\n{len(failed)} file(s) failed and were skipped: {[f[0] for f in failed]}")

    low_confidence = [r["source_file"] for r in records if r.get("diarization_confidence") == "low"]
    print(f"\nDone. {len(records)} call(s) transcribed.")
    if low_confidence:
        print(f"{len(low_confidence)} call(s) flagged low diarization confidence -- spot-check "
              f"these against the audio before trusting the speaker split: {low_confidence}")
    print(f"Wrote {out_json_path} (full transcripts, feeds 02_sentiment_analysis_gemini.py) and "
          f"{out_csv_path} (quick skim).")


if __name__ == "__main__":
    main()
