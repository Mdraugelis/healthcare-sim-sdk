"""Evaluator-side Markdown evidence packets; never import in the runner."""

import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any


ROLES = {
    "clinical": (
        "Clinical review", "Check population, usual care, treatment mechanisms, "
        "endpoint meaning and independent evidence for benefit or harm."),
    "operations": (
        "Operations review", "Check eligibility, assignment, staffing, capacity, "
        "adherence, resource-matched comparators and displaced care."),
    "statistics": (
        "Statistical review", "Check estimand, denominator, observation process, "
        "independent seeds, missing runs and the preregistered scoring rules."),
    "governance": (
        "Governance and equity review", "Check exposure history, target access, "
        "input release, subgroup scope and whether the intended claim is justified."),
    "engineering": (
        "Engineering review", "Check frozen source/config identities, isolated "
        "execution, complete run evidence and reproducibility."),
}


def _plain(value: Any) -> Any:
    """Preserve invalid evidence explicitly while writing strict JSON."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else {"invalid_numeric": str(number)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return {"unsupported_value": repr(value)}


def _json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, indent=2, allow_nan=False)


def _cell(value: Any) -> str:
    if value is None:
        return "undefined / not available"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: Any) -> str:
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{value:.4f}" if math.isfinite(float(value)) else "undefined"
    return _cell(value)


def render_report(
    output_dir: Any, case: dict, submission: dict, reference: dict, evaluation: dict,
) -> dict[str, str]:
    """Write an evaluator-only packet and preserve existing human notes verbatim.

    Packet identity binds all supplied evidence. It supports integrity, not
    confidentiality, and does not authenticate any source or human signature.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = {"case": case, "submission": submission, "reference": reference}
    identity = hashlib.sha256(
        _json({**records, "evaluation": evaluation}).encode()).hexdigest()
    paths: dict[str, str] = {}

    def save(key: str, filename: str, text: str) -> None:
        path = destination / filename
        path.write_text(text + "\n", encoding="utf-8")
        paths[key] = str(path.resolve())

    save("evaluation", "evaluation.json", _json(evaluation))
    save("records", "source_records.json", _json(records))
    human_path = destination / "human_review.json"
    if not human_path.exists():
        # Exclusive creation protects a reviewer writing while rendering starts.
        try:
            with human_path.open("x", encoding="utf-8") as handle:
                handle.write(_json({
                    "packet_identity": identity,
                    "roles": {
                        role: {"status": "NOT_RUN", "reviewer": "", "notes": "",
                               "reviewed_identity": "", "reviewed_at": ""}
                        for role in ROLES
                    },
                }) + "\n")
        except FileExistsError:
            pass
    paths["human_review"] = str(human_path.resolve())
    human_note = ("Human review is pending. Existing human notes, if any, "
                  "are preserved; the renderer does not authenticate or approve them.")
    try:
        prior = json.loads(human_path.read_text(encoding="utf-8"))
        if not isinstance(prior, dict) or prior.get("packet_identity") != identity:
            human_note += (" Existing notes do not identify this packet; "
                           "re-review is required.")
    except (OSError, ValueError):
        human_note += (" Existing notes could not be parsed; "
                       "they were left unchanged.")
    links = ("[Overview](review_packet.md) · [Evaluation](evaluation.json) · "
             "[Complete records](source_records.json) · "
             "[Human notes](human_review.json)")
    frozen = submission.get("identity") or evaluation.get("frozen_identity") or {}
    execution = submission.get("execution") or {}
    environment = execution.get("environment") or frozen.get("environment") or {}
    sources = frozen.get("source_hashes") or execution.get("source_hashes")
    code_digest = frozen.get("source_sha256")
    if sources is not None:
        serialized = json.dumps(_plain(sources), sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False)
        code_digest = hashlib.sha256(serialized.encode()).hexdigest()
    python = str(environment.get("python", "unavailable")).split()[0]
    numpy = environment.get("dependencies", {}).get("numpy", "unavailable")
    summary = [
        ("Case / study", f"{case.get('case_id')} / {case.get('study', {}).get('id')}"),
        ("Exposure / track",
         f"{case.get('exposure')} / {case.get('information_track')}"),
        ("Reference exposure", reference.get("exposure")),
        ("Session", frozen.get("session_id")),
        ("Case SHA-256", frozen.get("case_sha256") or execution.get("case_sha256")),
        ("Source map SHA-256 (canonical JSON)", code_digest),
        ("Runtime", f"Python {python}; NumPy {numpy}"),
        ("Submission SHA-256", submission.get("submission_sha256")),
        ("Execution mode / status",
         f"{execution.get('mode', 'unavailable')} / "
         f"{execution.get('status', 'unavailable')}"),
        ("Isolation self-test", execution.get("self_test", {}).get("status")),
        ("Packet SHA-256", identity),
    ]
    identity_lines = ["| Identity | Value |", "|---|---|",
                      *[f"| {label} | {_cell(value)} |" for label, value in summary],
                      "", "[Complete source identities and execution records]"
                      "(source_records.json) · [Frozen evaluator evidence]"
                      "(evaluation.json)", ""]
    statuses = ["| Dimension | Status |", "|---|---|"]
    for key in ("completeness", "numerical_fit", "blinding", "input_adequacy",
                "independent_validation"):
        statuses.append(f"| {key} | {_cell(evaluation.get(key, {}).get('status'))} |")
    if "protocol_integrity" in evaluation:
        value = evaluation["protocol_integrity"]
        value = value.get("status") if isinstance(value, dict) else value
        statuses.append(f"| protocol_integrity | {_cell(value)} |")
    scores = [
        "| Endpoint (units) | Reference | Prediction mean | Absolute error | CRPS | "
        "Baseline CRPS | Skill | Empirical interval | MCSE mean |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for eid, row in evaluation.get("endpoints", {}).items():
        interval = row.get("predictive_interval", {})
        values = [f"{eid} ({row.get('units')})", _number(row.get("reference_value")),
                  _number(row.get("mean")), _number(row.get("absolute_error")),
                  _number(row.get("crps")), _number(row.get("baseline_crps")),
                  _number(row.get("crps_skill")),
                  f"[{_number(interval.get('lower'))}, "
                  f"{_number(interval.get('upper'))}]", _number(row.get("mcse_mean"))]
        scores.append("| " + " | ".join(_cell(v) for v in values) + " |")
    if not evaluation.get("endpoints"):
        scores = ["Numerical scores were not produced. See failed checks; "
                  "no missing result is zero."]
    all_checks = evaluation.get("checks", [])
    counts: dict[str, int] = {}
    for item in all_checks:
        status = str(item.get("status", "MISSING"))
        counts[status] = counts.get(status, 0) + 1
    check_lines = [f"{len(all_checks)} checks recorded: " + ", ".join(
        f"{status} {count}" for status, count in sorted(counts.items())) + ".", ""]
    exceptions = [item for item in all_checks if item.get("status") != "PASS"]
    if exceptions:
        check_lines.extend([
            f"Showing {min(50, len(exceptions))} of {len(exceptions)} non-PASS checks. "
            "[All check records](evaluation.json).", "",
            "| Check | Status | Detail |", "|---|---|---|",
        ])
    else:
        check_lines.append("No non-PASS checks. [All check records](evaluation.json).")
    for item in exceptions[:50]:
        cells = [_cell(item.get(k)) for k in ("name", "status", "detail")]
        check_lines.append("| " + " | ".join(cells) + " |")
    notes = [f"- {note}" for note in evaluation.get("notes", [])]
    reasons = evaluation.get("input_adequacy", {}).get("reasons", [])
    source_lines = [f"- {reason}" for reason in reasons] or [
        "No provenance concerns recorded by this scorer; "
        "substantive adequacy still requires review."]
    all_runs = evaluation.get("retained_runs", [])
    run_lines = [f"{len(all_runs)} submitted run records retained. "
                 f"Showing the first {min(25, len(all_runs))} in submission order; "
                 "this table is a preview. [Complete inventory](evaluation.json) · "
                 "[Raw predictions and checks](source_records.json).", "",
                 "| Index | Seed | Recorded status |", "|---|---|---|"]
    for run in all_runs[:25]:
        cells = [_cell(run.get(k)) for k in ("index", "seed", "status")]
        run_lines.append("| " + " | ".join(cells) + " |")
    overview = [
        "# Benchmark evaluation review packet", "",
        "**Evaluator-only evidence: this packet includes observed targets. "
        "Do not place it in the runner's input bundle.**", "", human_note,
        "", links, "",
        "## Exact identities", "", *identity_lines,
        "## Separate findings", "", *statuses, "",
        "A numerical PASS means only that the preregistered numerical criteria passed. "
        "It is not independent validation, clinical adequacy or human approval.", "",
        "## Preregistered scoring", "", "```json",
        _json(case.get("scoring")), "```", "",
        "## Submitted predictions and observed statistics", "", *scores, "", *notes, "",
        "Values use the displayed endpoint units, rounded to four decimal places. "
        "Skill is dimensionless; full precision remains in evaluation.json.", "",
        "An interval containing this one observation is not evidence of "
        "calibrated coverage. Interval score also penalizes unnecessary width; "
        "its full value is in evaluation.json.", "",
        "## Input evidence", "", *source_lines, "",
        "Full used and unused source records, derivations and input values "
        "are in source_records.json.",
        "", "## Run inventory", "", *run_lines, "",
        "## Machine checks", "", *check_lines, "",
        "## Role-specific review", "",
        *[f"- [{title}](review_{role}.md)" for role, (title, _) in ROLES.items()],
    ]
    save("overview", "review_packet.md", "\n".join(overview))
    for role, (title, instruction) in ROLES.items():
        content = [f"# {title}", "", instruction, "", human_note, "", links, "",
                   "## Evidence identity", "", *identity_lines,
                   "## Findings requiring interpretation", "", *statuses, "",
                   "## Input provenance", "", *source_lines, "",
                   "## Review prompts", "",
                   "- Does the submitted statistic match the study population, "
                   "denominator, "
                   "time origin, assignment and estimator?",
                   "- Which inputs were genuinely available before outcomes, "
                   "and which were "
                   "assumed, learned from development, or explicitly released?",
                   "- What remains unsupported even if numerical criteria pass?",
                   "- Record your name, scope, limitations and this packet identity in "
                   "human_review.json. Machine output is not a signature.", "",
                   "Complete predictions, observed targets, run failures and "
                   "source dependencies are linked above. Role-specific "
                   "inspection supplements the common evidence."]
        save(f"review_{role}", f"review_{role}.md", "\n".join(content))
    return paths
