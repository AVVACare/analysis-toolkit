# Call Sentiment Pipeline (Gemini / Vertex AI)

Two-stage script pipeline over caregiver/navigator call recordings: transcribe via Gemini's native
audio understanding, then score sentiment/distress via a second Gemini call over the transcript.

This is a Gemini/GCP alternative to `../call-signal-pipeline/` (which uses local faster-whisper +
pyannote.audio + Kintsugi's DAM model, entirely on-Mac). Trade-offs versus that approach:

- No local model downloads, no `mamba`/conda environment split, no GPU/CPU acoustic model to run --
  just two Gemini calls per file, same shape as the extraction scripts in `../../home-assessment-forms/`.
- Speaker diarization is Gemini's own best-effort read of the audio (see `diarization_confidence` in
  the Stage 1 output), not a dedicated diarization model like `pyannote.audio`. Spot-check it.
- Sentiment/distress scoring comes from the transcript content itself (what was said), not acoustic
  voice biomarkers the way Kintsugi's DAM model works. These are genuinely different signals -- this
  pipeline doesn't replace DAM's acoustic read, it's a separate, complementary one.
- Calls audio confirmed mono, 8kHz telephony quality (via `ffprobe` on the sample files) -- Gemini
  reads that directly; no channel-splitting preprocessing needed either way.

## Setup

```bash
pip install -r requirements-gemini.txt
gcloud auth application-default login   # or a service account with Vertex AI User role
```

No Hugging Face account, no `mamba` environment, no model downloads -- auth is just your GCP
project having Vertex AI's Gemini API enabled.

## Running it — 2 stages, checkpointed

```bash
# Stage 1: transcribe every call in a directory
python 01_transcribe_gemini.py \
    --input-dir /path/to/Sample \
    --project-id avvacare-clinical-ops \
    --out-prefix ./transcripts \
    --limit 5   # start small, sanity-check transcripts.csv against a couple of real calls

# once transcripts.json looks right, run the rest (interrupted runs resume via work/<call>/transcript.json):
python 01_transcribe_gemini.py --input-dir /path/to/Sample --project-id avvacare-clinical-ops --out-prefix ./transcripts

# Stage 2: sentiment analysis over the transcripts
python 02_sentiment_analysis_gemini.py \
    --transcripts-json ./transcripts.json \
    --project-id avvacare-clinical-ops \
    --out-prefix ./call_sentiment
```

## What you get at the end

`transcripts.json` / `.csv` (Stage 1) -- per call: turn-by-turn transcript, speaker count, guessed
navigator speaker, `diarization_confidence`, audio quality issues.

`call_sentiment.json` / `.csv` (Stage 2) -- per call: `overall_sentiment` + numeric score,
`caregiver_distress_level`, `key_concerns` (confusion / overwhelm / decline / unmet_need /
financial_stress / safety_concern / caregiver_burnout / satisfaction / other), verbatim
`supporting_quotes` backing every rating, a one-paragraph `summary`, and `escalation_recommended`
for calls that warrant review sooner than routine cadence.

## Validating a non-English call

Some calls come in Armenian (and possibly other languages) -- if you don't speak the call's
language, you can't eyeball the raw transcript to check it. Both stages carry an English
translation alongside the original for exactly this reason:

- Stage 1: each turn has `text` (verbatim, original language) and `text_en` (English translation,
  null only when the call is already in English). `full_text_en()` in `01_transcribe_gemini.py`
  builds a fully English-readable transcript from it, and the CSV's `transcript_preview` column
  uses the translation automatically for any call flagged non-English (`transcript_preview_is_translated`
  tells you which).
- Stage 2: every entry in `supporting_quotes` is `{quote, quote_en}` -- the verbatim
  original-language quote plus its English translation, so you can verify a distress/concern rating
  against evidence you can actually read. `summary` is always written in English regardless of the
  call's language.

This doesn't replace a native speaker's review -- it's a translation from the same model doing the
scoring, not an independent check -- but it's enough to sanity-check whether the sentiment output is
in the right neighborhood before deciding whether a bilingual reviewer needs to look closer.

## Before trusting this at scale

Same caveat as everything else built this way so far: self-reported confidence
(`diarization_confidence`, `self_flagged_uncertain`) is useful context, not a substitute for spot
checks. Pull a sample of calls, read the transcripts and sentiment output against the actual audio,
and see where it holds up and where it doesn't -- the same validation step already applied to the
home-assessment extraction pipeline -- before wiring any of this into a CaReS factor.
