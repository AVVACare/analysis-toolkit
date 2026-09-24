#!/usr/bin/env python3
"""
Step 2 of the home-assessment analytics plan: prevalence and co-occurrence of the boolean risk
indicators in home_assessment.json.

Risk direction per field is set explicitly below rather than assumed -- a boolean's "risky" value
depends on the question wording (e.g. co_smoke_detectors_working=True is GOOD, patient_wandered_
before=True is BAD), so a naive "True = risk" default would mislabel half the safety fields. Two
fields are deliberately excluded from the risk set as context-only, not risk-directional on their
own:
  - home_safety.kitchen_access -- whether the patient has kitchen access isn't inherently a risk
    signal without more context (could reflect care plan, not hazard).
  - home_safety.patient_still_driving -- driving status alone isn't a risk without pairing it with
    a cognitive/behavioral flag; asserting a direction here would be a guess, not a finding.

Null handling: a field that's null (not applicable / not marked) is excluded from that field's own
prevalence rate (rate is % of non-null responses) and from any pair involving it in co-occurrence
(both fields must be non-null for a given patient to count in that pair).

Usage:
    python risk_prevalence_cooccurrence.py --input home_assessment.json --out-prefix ./risk

Outputs:
    {out-prefix}_prevalence.csv     -- one row per risk field: risk rate, counts, direction
    {out-prefix}_cooccurrence.csv   -- pairwise matrix: for each field pair, count of patients
                                        flagged at-risk on BOTH (among patients non-null on both)
    {out-prefix}_top_pairs.csv      -- same pairs, sorted by lift (co-occurrence vs. what
                                        independence would predict) and by raw count
    {out-prefix}_per_patient.csv    -- one row per patient: total risk-flag count + which fields
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

# field -> value that counts as "at risk". Comment shows the actual question text for traceability.
RISK_FIELDS = {
    # behavior.* -- True is always the concerning direction
    "behavior.agitation_aggression": True,
    "behavior.anxiety_shadowing": True,
    "behavior.hallucinations": True,
    "behavior.indifference_withdrawal": True,
    "behavior.irritability_moodiness": True,
    "behavior.sleep_problems": True,
    "behavior.suspiciousness_paranoia": True,
    # environmental_social.* -- True is always the concerning direction
    "environmental_social.fallen_in_last_year": True,
    "environmental_social.food_insecurity_risk": True,
    "environmental_social.housing_insecurity_risk": True,
    "environmental_social.socially_isolated": True,
    "environmental_social.targeted_for_exploitation": True,
    # home_safety.* -- direction depends on question wording; most are safety FEATURES
    # (True = good), so risk = False for those. patient_wandered_before is the one where
    # True itself is the risk.
    "home_safety.patient_wandered_before": True,
    "home_safety.navigates_home_independently": False,
    "home_safety.walkways_clear_of_hazards": False,
    "home_safety.stairs_clear_and_lit": False,
    "home_safety.bathroom_has_mobility_supports": False,
    "home_safety.stove_has_safety_features": False,
    "home_safety.medications_clearly_labeled": False,
    "home_safety.hazardous_items_securely_stored": False,
    "home_safety.firearms_securely_stored": False,
    "home_safety.co_smoke_detectors_working": False,
    "home_safety.fire_extinguisher_present": False,
    "home_safety.wandering_mitigations_in_place": False,
}


def get_nested(rec: dict, dotted_key: str):
    node = rec
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    records = json.loads(Path(args.input).read_text())
    n = len(records)
    fields = list(RISK_FIELDS.keys())

    # is_risk[field] = list of True/False/None (None = not applicable / null) per patient, in
    # record order
    is_risk = {f: [] for f in fields}
    for r in records:
        for f, risky_value in RISK_FIELDS.items():
            v = get_nested(r, f)
            is_risk[f].append(None if v is None else (v == risky_value))

    # --- prevalence ---
    prevalence_rows = []
    for f in fields:
        vals = is_risk[f]
        non_null = [v for v in vals if v is not None]
        risk_count = sum(1 for v in non_null if v)
        prevalence_rows.append({
            "field": f,
            "risky_value": RISK_FIELDS[f],
            "n_applicable": len(non_null),
            "n_not_applicable_null": n - len(non_null),
            "risk_count": risk_count,
            "risk_rate_pct_of_applicable": round(100 * risk_count / len(non_null), 1) if non_null else None,
        })
    prevalence_rows.sort(key=lambda r: (r["risk_rate_pct_of_applicable"] or 0), reverse=True)

    out_prev = Path(f"{args.out_prefix}_prevalence.csv")
    with out_prev.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prevalence_rows[0].keys()))
        w.writeheader()
        w.writerows(prevalence_rows)

    # --- pairwise co-occurrence + lift ---
    pair_rows = []
    for f1, f2 in itertools.combinations(fields, 2):
        v1, v2 = is_risk[f1], is_risk[f2]
        both_applicable = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
        if not both_applicable:
            continue
        n_both = len(both_applicable)
        both_risk = sum(1 for a, b in both_applicable if a and b)
        f1_risk_rate = sum(1 for a, _ in both_applicable if a) / n_both
        f2_risk_rate = sum(1 for _, b in both_applicable if b) / n_both
        expected = f1_risk_rate * f2_risk_rate * n_both
        lift = (both_risk / expected) if expected > 0 else None
        pair_rows.append({
            "field_1": f1,
            "field_2": f2,
            "n_both_applicable": n_both,
            "both_at_risk_count": both_risk,
            "both_at_risk_pct": round(100 * both_risk / n_both, 1),
            "lift_vs_independence": round(lift, 2) if lift is not None else None,
        })

    out_cooc = Path(f"{args.out_prefix}_cooccurrence.csv")
    with out_cooc.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    # Top pairs: require at least 4 co-occurring cases so lift isn't driven by 1-2 patients,
    # then rank by lift (surfaces combinations that cluster together more than chance) with raw
    # count as a tiebreaker.
    notable_pairs = [p for p in pair_rows if p["both_at_risk_count"] >= 4 and p["lift_vs_independence"]]
    notable_pairs.sort(key=lambda p: (p["lift_vs_independence"], p["both_at_risk_count"]), reverse=True)

    out_top = Path(f"{args.out_prefix}_top_pairs.csv")
    with out_top.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(notable_pairs)

    # --- per-patient composite ---
    per_patient_rows = []
    for i, r in enumerate(records):
        flagged = [f for f in fields if is_risk[f][i] is True]
        per_patient_rows.append({
            "source_file": r.get("source_file"),
            "patient_name": get_nested(r, "header.patient_name"),
            "risk_flag_count": len(flagged),
            "risk_fields_flagged": "; ".join(flagged),
        })
    per_patient_rows.sort(key=lambda r: r["risk_flag_count"], reverse=True)

    out_pp = Path(f"{args.out_prefix}_per_patient.csv")
    with out_pp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_patient_rows[0].keys()))
        w.writeheader()
        w.writerows(per_patient_rows)

    # --- console summary ---
    print(f"{n} patients, {len(fields)} risk fields evaluated.\n")
    print("Top 10 risk fields by prevalence (% of applicable/non-null forms flagged at-risk):")
    for row in prevalence_rows[:10]:
        print(f"  {row['field']:50s} {row['risk_rate_pct_of_applicable']:5.1f}%  "
              f"({row['risk_count']}/{row['n_applicable']} applicable, "
              f"{row['n_not_applicable_null']} null)")

    print("\nTop 10 co-occurring risk pairs by lift (>=4 shared cases):")
    for p in notable_pairs[:10]:
        print(f"  {p['field_1']} + {p['field_2']}: {p['both_at_risk_count']} patients "
              f"({p['both_at_risk_pct']}%), lift={p['lift_vs_independence']}x")

    dist = {}
    for row in per_patient_rows:
        dist[row["risk_flag_count"]] = dist.get(row["risk_flag_count"], 0) + 1
    print("\nRisk-flag-count distribution (per patient):")
    for k in sorted(dist):
        print(f"  {k} flags: {dist[k]} patient(s)")

    print(f"\nWrote {out_prev}, {out_cooc}, {out_top}, {out_pp}.")


if __name__ == "__main__":
    main()
