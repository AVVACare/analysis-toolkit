#!/usr/bin/env python3
"""
Stage 2: runs sentiment/distress analysis over the transcripts produced by
01_transcribe_gemini.py, via a text-only Gemini call per transcript.

Split into its own stage (rather than one combined audio-in-sentiment-out call) on purpose:
  - Transcription and sentiment are different jobs with different failure modes -- keeping them
    separate means a bad sentiment call doesn't force re-spending the (more expensive) audio call,
    and you can re-run sentiment alone if the rubric changes without re-transcribing anything.
  - Text-only calls are cheap and fast, so this stage can run at higher concurrency than Stage 1.
  - It mirrors the two-stage split already used in ../../home-assessment-forms/ (extract, then a
    separate comparison/scoring pass) rather than inventing a new pattern for this pipeline.

Scoring is evidence-based, not a bare label: every field that makes a judgment call
(caregiver_distress_level, key_concerns) must be backed by a verbatim quote in supporting_quotes,
same principle as the home-assessment work -- a score a reviewer can't trace back to the transcript
isn't trustworthy at face value, regardless of which model produced it.

Usage:
    python 02_sentiment_analysis_gemini.py \\
        --transcripts-json ./transcripts.json \\
        --project-id avvacare-clinical-ops \\
        --location us-central1 \\
        --out-prefix ./call_sentiment \\
        [--model gemini-2.5-flash] [--workers 8] [--limit 5]

Outputs:
    {out-prefix}.json -- one sentiment record per call, keyed by source_file
    {out-prefix}.csv  -- flattened, one row per call, for a quick skim / import into a report
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests
except ImportError:
    sys.exit("Missing dependency: pip install google-genai google-api-core")

RATE_LIMIT_ERRORS = (ResourceExhausted, TooManyRequests)

CONCERN_CATEGORIES = [
    "confusion", "overwhelm", "decline", "unmet_need", "financial_stress",
    "safety_concern", "caregiver_burnout", "satisfaction", "other",
]

SENTIMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overall_sentiment": {
            "type": "STRING", "enum": ["Positive", "Neutral", "Negative", "Mixed"],
            "description": "Overall emotional tone of the call.",
        },
        "overall_sentiment_score": {
            "type": "NUMBER",
            "description": "-1.0 (very negative) to 1.0 (very positive); 0 is neutral.",
        },
        "caregiver_distress_level": {
            "type": "STRING", "enum": ["None", "Mild", "Moderate", "Severe"],
            "description": "How much distress the PATIENT or FAMILY MEMBER (not the navigator) "
                           "expressed during the call -- frustration, fear, exhaustion, grief, "
                           "anger, hopelessness. 'None' if the call was routine/administrative.",
        },
        "per_speaker_sentiment": {
            "type": "ARRAY",
            "description": "One entry per distinct speaker label from the transcript.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "speaker": {"type": "STRING"},
                    "sentiment": {"type": "STRING", "enum": ["Positive", "Neutral", "Negative", "Mixed"]},
                    "notes": {"type": "STRING", "nullable": True},
                },
                "required": ["speaker", "sentiment"],
            },
        },
        "key_concerns": {
            "type": "ARRAY",
            "items": {"type": "STRING", "enum": CONCERN_CATEGORIES},
            "description": "Every category genuinely present in this call. Empty list if none.",
        },
        "supporting_quotes": {
            "type": "ARRAY",
            "description": "Verbatim quotes from the transcript that back up "
                           "caregiver_distress_level and key_concerns -- every non-trivial rating "
                           "above must be traceable to at least one quote here. Empty only if "
                           "overall_sentiment is Neutral and no concerns were flagged.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "quote": {"type": "STRING", "description": "Verbatim, in the language actually "
                                                                "spoken -- must match the transcript "
                                                                "text exactly, not a translation."},
                    "quote_en": {
                        "type": "STRING", "nullable": True,
                        "description": "English translation of `quote`. Required whenever the call "
                                       "is not in English -- this is what lets a reviewer who "
                                       "doesn't speak the call's language verify the rating against "
                                       "the evidence. Null only if the call is already in English.",
                    },
                },
                "required": ["quote"],
            },
        },
        "summary": {
            "type": "STRING",
            "description": "One-paragraph plain-language summary of what happened on the call and why "
                           "it was scored this way.",
        },
        "escalation_recommended": {
            "type": "BOOLEAN",
            "description": "True if a human should review this call soon rather than at routine "
                           "cadence -- e.g. safety concern, severe distress, or an unresolved urgent "
                           "need surfaced.",
        },
        "self_flagged_uncertain": {
            "type": "BOOLEAN",
            "description": "True if the transcript was too short, garbled, or ambiguous to score "
                           "confidently. Record this for context, but note -- per this project's own "
                           "prior finding on OCR/LLM self-confidence -- that this flag alone should "
                           "not be the only thing deciding whether a call gets human review.",
        },
    },
    "required": ["overall_sentiment", "overall_sentiment_score", "caregiver_distress_level",
                 "key_concerns", "supporting_quotes", "summary", "escalation_recommended"],
}

PROMPT_TEMPLATE = """You are analyzing the sentiment and distress signals in a transcript of a phone
call between an AvvaCare caregiver navigator and a patient or family member. The transcript below
was produced by an automated speech-to-text pass and may contain minor transcription errors or
uncertain speaker labels -- work with it as given.

Score the PATIENT/FAMILY MEMBER's emotional state, not the navigator's (the navigator is AvvaCare
staff conducting a routine check-in call; their tone is not the signal of interest). Every rating
of caregiver_distress_level above "None", and every entry in key_concerns, must be backed by at
least one verbatim quote in supporting_quotes -- do not assert a concern or distress level you
cannot point to in the transcript text.

Flag escalation_recommended = true for anything suggesting a safety risk, an urgent unmet need, or
severe emotional distress that shouldn't wait for the next routine review cycle.

Write `summary` in English regardless of what language the call was in. If the call is not in
English, every entry in `supporting_quotes` must include both the verbatim original-language
`quote` and an English `quote_en` translation -- someone reviewing this output may not speak the
call's language and needs to verify your rating against the evidence either way.

Transcript ({num_turns} turns, source file: {source_file}):
---
{transcript_text}
---
"""


def full_transcript_text(record: dict) -> str:
    return "\n".join(f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in record.get("turns", []))


def analyze_one_call(client, model: str, record: dict, schema: dict) -> dict:
    transcript_text = full_transcript_text(record)
    prompt = PROMPT_TEMPLATE.format(
        num_turns=len(record.get("turns", [])),
        source_file=record.get("source_file", "unknown"),
        transcript_text=transcript_text,
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = json.loads(response.text)
    parsed["source_file"] = record.get("source_file")
    return parsed


def analyze_with_retry(client, model, record, schema, max_attempts=5) -> dict:
    for attempt in range(1, max_attempts + 1):
        try:
            return analyze_one_call(client, model, record, schema)
        except RATE_LIMIT_ERRORS:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)


def format_quote(q: dict) -> str:
    quote, quote_en = q.get("quote", ""), q.get("quote_en")
    return f'{quote} [EN: {quote_en}]' if quote_en else quote


def flatten_for_csv(record: dict) -> dict:
    return {
        "source_file": record.get("source_file"),
        "overall_sentiment": record.get("overall_sentiment"),
        "overall_sentiment_score": record.get("overall_sentiment_score"),
        "caregiver_distress_level": record.get("caregiver_distress_level"),
        "key_concerns": "; ".join(record.get("key_concerns", []) or []),
        "escalation_recommended": record.get("escalation_recommended"),
        "self_flagged_uncertain": record.get("self_flagged_uncertain"),
        "summary": record.get("summary"),
        "supporting_quotes": " | ".join(format_quote(q) for q in record.get("supporting_quotes", []) or []),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts-json", required=True,
                     help="Output of 01_transcribe_gemini.py (the {out-prefix}.json file).")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--workers", type=int, default=8,
                     help="Text-only calls are cheap/fast relative to Stage 1's audio calls, so "
                          "this can run at higher concurrency by default.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    transcripts_path = Path(args.transcripts_json)
    if not transcripts_path.exists():
        sys.exit(f"{transcripts_path} not found -- run 01_transcribe_gemini.py first.")
    all_records = json.loads(transcripts_path.read_text())
    if args.limit:
        all_records = all_records[: args.limit]
    if not all_records:
        sys.exit(f"No transcript records found in {transcripts_path}")

    empty_calls = [r["source_file"] for r in all_records if not r.get("turns")]
    scoreable = [r for r in all_records if r.get("turns")]
    if empty_calls:
        print(f"Skipping {len(empty_calls)} call(s) with no transcribed turns (empty/failed "
              f"transcription upstream): {empty_calls}")

    client = genai.Client(vertexai=True, project=args.project_id, location=args.location)

    print(f"Scoring sentiment for {len(scoreable)} call(s) ({args.workers} concurrent)...")

    results = []
    failed = []
    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_with_retry, client, args.model, r, SENTIMENT_SCHEMA): r for r in scoreable}
        for future in as_completed(futures):
            record = futures[future]
            done_count += 1
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{done_count}/{len(scoreable)}] {record.get('source_file')} ... FAILED ({e})")
                failed.append((record.get("source_file"), str(e)))
                continue
            flag = " -- ESCALATE" if result.get("escalation_recommended") else ""
            print(f"[{done_count}/{len(scoreable)}] {record.get('source_file')} ... "
                  f"{result.get('overall_sentiment')} / distress={result.get('caregiver_distress_level')}{flag}")
            results.append(result)

    order = {r["source_file"]: i for i, r in enumerate(all_records)}
    results.sort(key=lambda r: order.get(r["source_file"], 999999))

    out_json_path = Path(f"{args.out_prefix}.json")
    out_csv_path = Path(f"{args.out_prefix}.csv")
    out_json_path.write_text(json.dumps(results, indent=2))

    flat_records = [flatten_for_csv(r) for r in results]
    fieldnames = list(flat_records[0].keys()) if flat_records else []
    with out_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_records)

    if failed:
        print(f"\n{len(failed)} call(s) failed and were skipped: {[f[0] for f in failed]}")

    escalations = [r["source_file"] for r in results if r.get("escalation_recommended")]
    print(f"\nDone. {len(results)} call(s) scored.")
    if escalations:
        print(f"{len(escalations)} call(s) flagged for escalation -- review these first: {escalations}")
    print(f"Wrote {out_json_path} and {out_csv_path}.")


if __name__ == "__main__":
    main()
