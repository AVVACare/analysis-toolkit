# Data-Driven Care Insights: Executive Summary

*Home Assessments & Call Intelligence — September 2026*

## Executive Summary

- 100 handwritten home-assessment forms were turned into structured, analyzable patient data.
- Clear, quantifiable risk patterns emerged across the patient population once the data was combined.
- A pilot system was built to automatically transcribe and analyze caregiver check-in calls.

## Home Assessments

### What we did

AvvaCare's In-Home Assessment forms are filled out by hand during every caregiver visit, capturing home safety, environment, and behavioral health. We built an automated system that reads these scanned forms and turns them into structured, searchable data — 100 forms processed so far, spanning 24 distinct safety and behavioral risk indicators per patient.

### Why Gemini

We evaluated three ways to read the scanned forms: Google Document AI, Google Cloud Vision OCR, and Gemini. Document AI was ruled out early — its table parser couldn't correctly separate the Daily Function section's rating columns, losing which answer was actually marked on the form.

To choose between the other two, we manually checked 754 individual answers, across 38 different questions, against the original scans. **Gemini matched the correct answer 87% of the time; Cloud Vision matched only 3% of the time** — Cloud Vision reads raw text rather than identifying which specific answer was marked, so it wasn't built for this kind of structured reading in the first place. Based on this, Gemini became the primary extraction method, with any result below an 85% reliability bar routed to manual review rather than trusted automatically.

### Finding 1 — A widespread, fixable gap

![Top 12 risk indicators by prevalence](chart_prevalence.png)

**93% of homes lack basic stove safety features** (safety knobs or an automatic shut-off) — the single most common pattern found across the entire group, affecting nearly every home rather than a handful of cases. Because it's so widespread, it doesn't help distinguish one patient's risk from another's, but it does point to a systemic opportunity worth addressing at scale.

The behavior-symptom indicators (agitation, irritability, anxiety, sleep problems, withdrawal) each appear in roughly 40–50% of forms — common enough to matter, but varied enough across patients to actually be useful for telling them apart.

### Finding 2 — Patterns that cluster together

![Co-occurring risk pairs: rate with vs. without the first flag](chart_cooccurrence.png)

Some risk indicators show up together far more often than they would if they were unrelated. Each pair below compares two rates: how often the second flag appears among patients who already have the first flag, versus how often it appears across the whole patient population. Pairs with fewer than 15 qualifying patients are left out, since a small group can produce a misleadingly large gap.

**Among patients with withdrawal, 59.5% also show social isolation, compared to 28.7% across all patients** — the largest such pattern by patient count (22 of 100). **Among patients with walkway hazards, 53.3% also have missing bathroom mobility supports, compared to 21.9% across all patients** — an even bigger gap in percentage-point terms, though in a smaller group (8 patients). A broader behavior-symptom cluster (hallucinations, paranoia, agitation, wandering, reduced independent navigation) shows the same pattern, each holding across 34–47 patients.

These are the kinds of patterns that are easy to miss reading forms one at a time, and easy to see once the data is combined.

### Finding 3 — Patients aren't all alike

![Patients by total risk-flag count](chart_distribution.png)

Each patient's assessment was scored against 24 possible risk indicators. The spread shown above — most patients clustered in the middle, with some much higher or lower — is what makes a composite count like this useful: it can meaningfully separate patients from one another, rather than labeling everyone the same way.

## Call Intelligence

### What we did

We also piloted a system that automatically transcribes and analyzes caregiver check-in calls with patients and families — flagging emotional distress, key concerns, and calls that need urgent follow-up, without anyone having to listen to every recording. It even works across languages: non-English calls are translated alongside the original so any reviewer can verify what was flagged and why.

### Early results (pilot stage)

- **Strong initial quality** — tested on an initial set of sample calls, with accurate transcription and sentiment scoring.
- **Automatic concern detection** — flags distress level and specific concerns (e.g. financial stress, safety, caregiver burnout), plus calls needing urgent review.
- **Works across languages** — non-English calls are translated for easy verification by any reviewer.

This is still pilot-stage: the next step is validating the pipeline on a larger batch of calls before broader rollout.

## Where We Go From Here

1. Expand the home-assessment pipeline as new forms come in.
2. Validate the call-intelligence pipeline on a larger batch of calls.
3. Use these structured signals to support care prioritization going forward.
