"""Run-specific evidence and role-specific human review packets.

This is an experiment consumer; it does not change the scenario contract.
Machine evidence never supplies human approval. Missing evidence remains
visible, even when all checks that happened to run passed.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from numbers import Integral, Real
from pathlib import Path
import platform
import subprocess
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from .validate import load_json_object


STATUSES = {"PASS", "FAIL", "NOT_RUN", "INCONCLUSIVE", "NOT_APPLICABLE"}
ROLES = {
    "clinical": ("Clinical review", "Check the population, clinical mechanisms, "
                 "usual care, effect assumptions, and helped/harmed cases."),
    "operations": ("Operations review", "Check eligibility, staffing, capacity, "
                   "queues, adoption, and who is displaced."),
    "statistics": ("Statistical review", "Check the estimand, denominators, "
                   "comparison, uncertainty, missing runs, and study design."),
    "governance": ("Governance and equity review", "Check subgroup coverage, "
                   "benefit and burden, missing evidence, and intended use."),
    "engineering": ("Engineering review", "Check code/config/model identity, "
                    "reproduction, invariants, traces, and artifact integrity."),
}
REQUIRED_PACKET_ARTIFACTS = {
    "config.json", "registry.json", "evidence.json", "review_packet.md",
    "case_appendix.md",
    *(f"review_{role}.md" for role in ROLES),
}
RESERVED_ARTIFACTS = REQUIRED_PACKET_ARTIFACTS | {"manifest.json", "human_review.json"}


def _run_key(cell, seed):
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Run seed must be a nonnegative integer")
    return str(cell), int(seed)


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def canonical_json(value) -> str:
    return json.dumps(_plain(value), sort_keys=True, allow_nan=False,
                      ensure_ascii=False, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root, *args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_identity(config, model_path=None, source_root=None):
    """Hash source contents, resolved values, model bytes and environment.

    Git-ignored institutional files are excluded from source discovery;
    a supplied model is fingerprinted explicitly, not copied into source.
    """
    root = Path(source_root or Path(__file__).resolve().parents[2])
    files = _git(root, "ls-files", "-z", "--cached", "--others",
                 "--exclude-standard")
    names = files.split("\0") if files else [
        str(p.relative_to(root)) for p in root.glob("healthcare_sim_sdk/**/*.py")
    ]
    relevant = sorted(set(n for n in names if n and (
        n.startswith(("healthcare_sim_sdk/", "scripts/", "tests/"))
        or n in ("pyproject.toml", ".claude/invariants.yaml")
    ) and Path(n).suffix in (".py", ".yaml", ".yml", ".toml")))
    sources = {n: file_digest(root / n) if (root / n).is_file() else "DELETED"
               for n in relevant}
    dependencies = {}
    for name in ("numpy", "scipy", "hydra-core", "omegaconf", "statsmodels"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not installed"
    identity = {
        "revision": _git(root, "rev-parse", "HEAD"),
        "source_sha256": digest(sources),
        "config_sha256": digest(config),
        "model_sha256": file_digest(model_path) if model_path else None,
        "python": platform.python_version(), "dependencies": dependencies,
    }
    identity["identity_sha256"] = digest(identity)
    return identity


def evidence(name, status, *, detail="", actual=None, expected=None,
             category="simulation", required=True, rationale="", artifacts=()):
    if not isinstance(required, bool):
        raise ValueError("Evidence required must be a boolean")
    status = str(getattr(status, "value", status))
    if status not in STATUSES:
        raise ValueError(f"Unknown evidence status: {status}")
    if status == "NOT_APPLICABLE" and (
        not isinstance(rationale, str) or not rationale.strip()
    ):
        raise ValueError("NOT_APPLICABLE requires a rationale")
    return {
        "name": name, "status": status, "detail": detail,
        "actual": _plain(actual), "expected": _plain(expected),
        "category": category, "required": required, "rationale": rationale,
        "artifacts": list(artifacts),
    }


def normalize_check(check):
    c = _plain(check)
    if not isinstance(c, dict):
        raise ValueError("Evidence must be a check mapping")
    status = c.get("status")
    if status is None:
        legacy = c.get("pass", c.get("passed", False))
        if not isinstance(legacy, bool):
            raise ValueError("Legacy evidence pass/passed must be boolean")
        status = "PASS" if legacy else "FAIL"
    return evidence(
        c.get("name", c.get("check", c.get("boundary", "unnamed"))),
        status,
        detail=c.get("detail", ""), actual=c.get("actual"),
        expected=c.get("expected"), category=c.get("category", "simulation"),
        required=c.get("required", True), rationale=c.get("rationale", ""),
        artifacts=c.get("artifacts", []),
    )


def aggregate_status(checks):
    if not checks:
        return "NOT_RUN"
    checks = [normalize_check(check) for check in checks]
    statuses = [c["status"] for c in checks if c.get("required", True)]
    if any(c["status"] == "FAIL" for c in checks):
        return "FAIL"
    for status in ("NOT_RUN", "INCONCLUSIVE"):
        if status in statuses:
            return status
    return "PASS" if statuses else "NOT_RUN"


def load_registry(path):
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return validate_registry(raw)


def validate_registry(raw):
    if not isinstance(raw, dict) or not raw.get("decision_question"):
        raise ValueError("Registry needs a decision_question")
    claims = raw.get("claims", [])
    ids = [c.get("id") for c in claims]
    if not ids or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError("Registry needs unique, nonempty claim IDs")
    for c in claims:
        for key in ("question", "estimand", "population", "denominator",
                    "comparison", "horizon", "acceptance", "reviewers", "evidence"):
            if not c.get(key):
                raise ValueError(f"Claim {c['id']} missing {key}")
        if any(r not in ROLES for r in c["reviewers"]):
            raise ValueError(f"Unknown reviewer role in {c['id']}")
        if c.get("scope", "run") not in ("run", "experiment"):
            raise ValueError("Claim scope must be run or experiment")
    return raw


class ReviewPacket:
    """Accumulate actual cell/seed evidence and render a review dossier."""

    def __init__(self, output_dir, registry, config, expected_runs,
                 model_path=None, source_root=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = (load_registry(registry) if not isinstance(registry, dict)
                         else validate_registry(registry))
        self.config = _plain(config)
        self.identity = run_identity(config, model_path, source_root)
        self.identity["registry_sha256"] = digest(self.registry)
        self._identity_parameters = (self.config, model_path, source_root)
        self.expected = [_run_key(cell, seed) for cell, seed in expected_runs]
        if not self.expected or len(set(self.expected)) != len(self.expected):
            raise ValueError("Expected cell/seed pairs must be nonempty and unique")
        self.identity["expected_runs_sha256"] = digest(self.expected)
        self.identity["identity_sha256"] = digest({
            key: value for key, value in self.identity.items()
            if key != "identity_sha256"
        })
        self.runs = {}
        self.experiment_checks = []
        self._certificates = {}
        self._artifact_hashes = {}
        self._extra_artifacts = set()

    def add_run(self, cell, seed, *, metrics=None, checks=(), artifacts=(),
                cases=None, error=None):
        key = _run_key(cell, seed)
        if key not in self.expected or key in self.runs:
            raise ValueError(f"Unexpected or duplicate run: {key}")
        normalized = [normalize_check(c) for c in checks]
        if len({c["name"] for c in normalized}) != len(normalized):
            raise ValueError("Duplicate evidence names in a run")
        if error:
            normalized.append(evidence("execution", "FAIL", detail=str(error)))
        artifact_names = {str(name) for name in artifacts}
        artifact_names.update(
            str(name) for check in normalized for name in check["artifacts"])
        artifact_names.update(
            str(case["artifact"]) for case in (cases or []) if case.get("artifact"))
        for name in artifact_names:
            self._remember_artifact(name)
        for metric in self.registry.get("metrics", []):
            value = (metrics or {}).get(metric["key"])
            missing = value is None
            numeric = isinstance(value, Real) and not isinstance(value, bool)
            normalized.append(evidence(
                f"recorded_metric:{metric['key']}",
                "NOT_RUN" if missing else "PASS" if numeric else "FAIL",
                detail=(
                    "Required registry metric must be present as a "
                    "finite scalar number."
                ),
                category="analysis", actual=value,
                required=metric.get("required", True),
            ))
        record = {"cell": key[0], "seed": key[1], "metrics": metrics or {},
                  "checks": normalized, "cases": cases or [],
                  "artifacts": sorted(artifact_names), "error": error}
        canonical_json(record)  # reject invalid evidence before recording it
        self.runs[key] = _plain(record)

    def add_evidence(self, check):
        c = normalize_check(check)
        if any(x["name"] == c["name"] for x in self.experiment_checks):
            raise ValueError("Duplicate experiment evidence name")
        for name in c["artifacts"]:
            self._remember_artifact(name)
        self.experiment_checks.append(c)

    def _remember_artifact(self, name):
        path = _artifact_path(self.output_dir, name)
        if name in RESERVED_ARTIFACTS:
            raise ValueError(f"Artifact name reserved for generated content: {name}")
        if not path.is_file():
            raise ValueError(f"Missing evidence artifact: {name}")
        checksum = file_digest(path)
        previous = self._artifact_hashes.get(name)
        if previous is not None and previous != checksum:
            raise ValueError(f"Evidence artifact changed after recording: {name}")
        self._artifact_hashes[name] = checksum

    def add_artifact(self, name):
        """Bind an experiment-level artifact inside the packet to its bytes."""
        name = str(name)
        self._remember_artifact(name)
        self._extra_artifacts.add(name)

    def add_certificate(self, name, path, expected_identity=None):
        """Reuse a certificate only with an exact identity and artifact match."""
        path = Path(path)
        if not path.exists():
            self.add_evidence(evidence(name, "NOT_RUN", detail="Certificate absent"))
            return
        try:
            cert = load_json_object(path)
            if (not isinstance(cert.get("checks"), list)
                    or not isinstance(cert.get("artifacts"), dict)):
                raise ValueError((
                    "Certificate needs a check list and artifact hash "
                    "mapping"
                ))
            normalized = [normalize_check(c) for c in cert["checks"]]
            if len({c["name"] for c in normalized}) != len(normalized):
                raise ValueError("Duplicate certificate checks")
            identity = expected_identity or self.identity
            valid = cert.get("identity") == identity
            valid &= bool(normalized)
            valid &= aggregate_status(normalized) == "PASS"
            valid &= bool(cert.get("artifacts")) and all(
                _artifact_path(path.parent, a).is_file()
                and file_digest(_artifact_path(path.parent, a)) == h
                for a, h in cert.get("artifacts", {}).items())
            status = "PASS" if valid else "INCONCLUSIVE"
            detail = "Identity and artifacts match" if valid else (
                "Stale, incompatible, incomplete, or failed "
                "certificate"
            )
            if any(c["status"] == "FAIL" for c in normalized):
                status, detail = "FAIL", "Certificate contains failed checks"
            if valid:
                self._certificates[name] = {
                    "path": path, "sha256": file_digest(path),
                    "artifacts": dict(cert["artifacts"]),
                }
        except (ValueError, TypeError, KeyError, OSError):
            status, detail = "FAIL", "Unreadable certificate"
        proof = None
        if name in self._certificates:
            proof = {"certificate_sha256": self._certificates[name]["sha256"],
                     "certified_identity": cert["identity"],
                     "artifact_hashes": cert["artifacts"]}
        self.add_evidence(evidence(
            name, status, detail=detail, actual=proof, category="certification"))

    def _claims(self):
        claims = []
        for claim in self.registry["claims"]:
            checks = []
            sources = (
                [self.experiment_checks]
                if claim.get("scope", "run") == "experiment"
                else [self.runs.get(key, {}).get("checks", []) for key in self.expected]
            )
            for source in sources:
                by_name = {c["name"]: c for c in source}
                for name in claim["evidence"]:
                    check = dict(by_name.get(name, evidence(
                        name, "NOT_RUN", detail="Required evidence absent")))
                    # Registry requirements override optional diagnostic flags.
                    check["required"] = True
                    checks.append(check)
            claims.append({**claim, "status": aggregate_status(checks),
                           "evidence_results": checks})
        return claims

    def write(self):
        """Write raw evidence first, then role views and an integrity manifest."""
        current_identity = run_identity(*self._identity_parameters)
        stable = all(
            self.identity.get(key) == value
            for key, value in current_identity.items() if key != "identity_sha256"
        ) and digest(self.registry) == self.identity["registry_sha256"]
        provenance = evidence("packet_provenance", "PASS" if stable else "INCONCLUSIVE",
                              category="provenance", detail=(
                                  "Code, configuration, model and registry checked "
                                  "at write time"
                              ))
        self.experiment_checks = [
            c for c in self.experiment_checks if c["name"] != "packet_provenance"]
        self.experiment_checks.append(provenance)
        for name, checksum in self._artifact_hashes.items():
            path = _artifact_path(self.output_dir, name)
            if not path.is_file() or file_digest(path) != checksum:
                raise ValueError(f"Evidence artifact changed after recording: {name}")
        for name, record in self._certificates.items():
            path = record["path"]
            valid = path.is_file() and file_digest(path) == record["sha256"]
            valid &= all(
                _artifact_path(path.parent, artifact).is_file()
                and file_digest(_artifact_path(path.parent, artifact)) == checksum
                for artifact, checksum in record["artifacts"].items())
            if not valid:
                check = next(c for c in self.experiment_checks if c["name"] == name)
                check.update(status="INCONCLUSIVE", detail=(
                    "Certificate or its evidence changed after "
                    "recording"
                ))
        claims = self._claims()
        missing = [list(k) for k in self.expected if k not in self.runs]
        checks = ([c for run in self.runs.values() for c in run["checks"]]
                  + self.experiment_checks)
        status = aggregate_status(
            [evidence(c["id"], c["status"]) for c in claims] + checks)
        if missing and status != "FAIL":
            status = "NOT_RUN"
        files = []

        def save(name, data):
            path = self.output_dir / name
            path.write_text(data)
            files.append(name)

        save("config.json", json.dumps(self.config, indent=2, allow_nan=False))
        save("registry.json", json.dumps(self.registry, indent=2, allow_nan=False))
        save("evidence.json", json.dumps({
            "identity": self.identity, "claims": claims,
            "runs": list(self.runs.values()),
            "experiment_checks": self.experiment_checks,
        }, indent=2, allow_nan=False))
        save("case_appendix.md", "\n".join([
            "# Complete case appendix", "",
            "Every recorded run and category is retained below, including absences.",
            "[Packet overview](review_packet.md) · [Source records](evidence.json)",
            "", *self._case_markdown(concise=False),
        ]))
        human_path = self.output_dir / "human_review.json"
        if not human_path.exists():
            human_path.write_text(json.dumps({
                r: {"status": "NOT_RUN", "reviewer": "", "reviewed_identity": "",
                    "notes": "", "reviewed_at": ""}
                for r in ROLES
            }, indent=2))
        # Human notes are intentionally mutable and excluded from machine hashes.
        for role, (title, instruction) in ROLES.items():
            selected = [c for c in claims if role in c["reviewers"]]
            lines = [f"# {title}", "", instruction, "",
                     (
                         "Human review is pending; machine checks do not "
                         "approve deployment."
                     ), "",
                     (
                         "[Packet overview](review_packet.md) · "
                         "[Evidence](evidence.json) · [Review "
                         "notes](human_review.json)"
                     ), ""]
            for c in selected:
                lines.extend(_claim_markdown(c))
            lines.extend(["## Assumptions to review", ""])
            assumptions = [a for a in self.registry.get("assumptions", [])
                           if role in a.get("reviewers", list(ROLES))]
            for a in assumptions:
                lines.append(
                    f"- **{a.get('id', 'assumption')}**: {a.get('description', '')} "
                    f"Source: {a.get('source', 'not supplied')}; "
                    f"provenance: {a.get('provenance', 'unknown')}; "
                    f"range: {a.get('range', 'not supplied')}; "
                    f"owner: {a.get('owner', 'unassigned')}.")
            lines.extend(["", "## Cases and reproduction", "",
                          (
                              "Case records in evidence.json identify their "
                              "cell, seed, selection rule, and trace artifacts."
                          ),
                          (
                              "Unavailable case categories and unrun checks "
                              "remain explicit. Verify these against the "
                              "reviewed configuration."
                          ), ""])
            lines.extend(self._results_markdown())
            if role in ("clinical", "operations", "governance"):
                lines.extend(self._case_markdown())
            else:
                lines.extend([
                    "[Inspect every run and case category](case_appendix.md)", ""])
            save(f"review_{role}.md", "\n".join(lines))
        lines = ["# Simulation review packet", "",
                 f"**Decision question:** {self.registry['decision_question']}", "",
                 f"**Machine evidence:** {status} · **Human review:** pending", "",
                 f"**Run identity:** `{self.identity['identity_sha256']}`", "",
                 f"**Completeness:** {len(self.runs)}/{len(self.expected)} "
                 f"cell/seed runs recorded; {len(missing)} missing.", "",
                 (
                     "Results are conditional on the configured "
                     "assumptions. PASS describes evidence, not the "
                     "direction of benefit or clinical effectiveness."
                 ), "",
                 "## Choose your review", ""]
        lines += [f"- [{title}](review_{role}.md)"
                  for role, (title, _) in ROLES.items()]
        lines += ["", "## Claims", "", "| Claim | Status | Reviewers |",
                  "|---|---|---|"]
        lines += [f"| {c['question']} | {c['status']} | {', '.join(c['reviewers'])} |"
                  for c in claims]
        lines += ["", "## Unresolved evidence", ""]
        unresolved = [(c["id"], e) for c in claims for e in c["evidence_results"]
                      if e["status"] not in ("PASS", "NOT_APPLICABLE")]
        lines += sorted(set(
            f"- {name}: {e['name']} — {e['status']}. {e['detail']}"
            for name, e in unresolved)) or [(
                "No missing required machine evidence in the "
                "declared claims. Human review remains pending."
            )]
        if missing:
            lines.append(f"- Missing cell/seed pairs: {missing}")
        lines.extend(["", "## Observed simulation results", ""])
        lines.extend(self._results_markdown())
        lines.extend(self._case_markdown())
        lines += ["", "## Evidence and limits", "",
                  (
                      "[Manifest](manifest.json) · [Resolved "
                      "config](config.json) · [Claims and "
                      "assumptions](registry.json) · [Machine "
                      "evidence](evidence.json)"
                  ), "",
                  (
                      "Seed distribution intervals, Monte Carlo mean "
                      "intervals, study-estimator intervals, and "
                      "input-assumption uncertainty are distinct."
                  ),
                  (
                      "Required checks that were not run are not passes. "
                      "Historical or oracle evidence is not "
                      "certification of an observable-data study."
                  ), ""]
        if self._extra_artifacts:
            lines.extend(["## Experiment analysis artifacts", ""])
            lines.extend(f"- [{name}]({name})"
                         for name in sorted(self._extra_artifacts))
            lines.append("")
        save("review_packet.md", "\n".join(lines))
        artifacts = {name: file_digest(self.output_dir / name) for name in files}
        artifacts.update(self._artifact_hashes)
        for run in self.runs.values():
            for name in run["artifacts"]:
                path = _artifact_path(self.output_dir, name)
                if not path.is_file():
                    raise ValueError(f"Missing run artifact: {name}")
                artifacts[name] = file_digest(path)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": self.identity,
            "expected_runs": [list(k) for k in self.expected],
            "completed_runs": [list(k) for k in self.runs], "missing_runs": missing,
            "failed_runs": [list(k) for k, r in self.runs.items()
                            if aggregate_status(r["checks"]) == "FAIL"],
            "machine_status": status, "human_review": "NOT_RUN",
            "artifacts": artifacts,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False))
        return self.output_dir / "review_packet.md"

    def _results_markdown(self):
        from .uncertainty import summarize_values
        lines = [(
            "| Cell | Metric (unit) | Seeds | Mean | Seed "
            "2.5–97.5% range | MCSE of mean |"
        ),
                 "|---|---|---|---|---|---|"]
        for cell in dict.fromkeys(k[0] for k in self.expected):
            label = next((item.get("label", cell)
                          for item in self.config.get("resolved_cells", [])
                          if item.get("cell") == cell), cell)
            display_cell = f"{label} ({cell})" if label != cell else cell
            runs = [r for k, r in self.runs.items() if k[0] == cell]
            expected_count = sum(key[0] == cell for key in self.expected)
            for metric in self.registry.get("metrics", []):
                values = [r["metrics"].get(metric["key"]) for r in runs]
                if (len(runs) != expected_count or not values
                        or any(not isinstance(value, Real) or isinstance(value, bool)
                               for value in values)):
                    lines.append(
                        f"| {display_cell} | {metric['key']} | "
                        "incomplete | — | — | — |")
                    continue
                s = summarize_values(values)
                mcse = (f"{s['mcse_mean']:.4g}" if s["mcse_mean"] is not None
                        else "INCONCLUSIVE (one seed)")
                lines.append(
                    f"| {display_cell} | {metric.get('label', metric['key'])} "
                    f"({metric.get('unit', '')}) | {s['n_seeds']} | {s['mean']:.4g} "
                    f"| {s['seed_p025']:.4g} to {s['seed_p975']:.4g} | {mcse} |")
        return lines + ["", (
            "These are simulated outcomes conditional on "
            "inputs; the seed range is not a confidence "
            "interval for the mean."
        ), ""]

    def _case_markdown(self, concise=True):
        lines = ["### Cases from recorded runs", ""]
        records = [(run, case) for run in self.runs.values() for case in run["cases"]]
        absences = {}
        if concise:
            lines.extend([
                "Highlights show the first available matching case per category. "
                "They are illustrative; each case retains its original selection rule.",
                "[Every run/category, including unavailable cases](case_appendix.md)",
                "",
            ])
            selected = {}
            for run, case in records:
                category = case.get("category", "case")
                if (category not in selected or
                        (selected[category][1].get("status") != "PASS"
                         and case.get("status") == "PASS")):
                    selected[category] = (run, case)
                if case.get("status") != "PASS":
                    absences[category] = absences.get(category, 0) + 1
            records = list(selected.values())
        for run, case in records:
            category = case.get("category", "case")
            lines.append(
                f"- **{run['cell']}, seed {run['seed']} — {category}:** "
                f"{case.get('status', 'NOT_RUN')}. {case.get('narrative', '')} "
                f"Selection: {case.get('selection', 'unspecified')}; "
                f"candidates: {case.get('candidate_count', 'unknown')}.")
            if concise and absences.get(category):
                lines.append(
                    f"  Unavailable/unresolved in {absences[category]} "
                    "recorded run(s); "
                    "see the complete appendix.")
            if case.get("artifact"):
                lines.append(f"  [Underlying trace]({case['artifact']})")
        return lines + [""]


def _claim_markdown(claim):
    lines = [f"## {claim['question']}", "",
             f"**Evidence status:** {claim['status']}", ""]
    for key in ("estimand", "population", "denominator", "comparison", "horizon",
                "acceptance"):
        lines.append(f"- **{key.capitalize()}:** {claim[key]}")
    lines += ["", "| Evidence | Status | Detail |", "|---|---|---|"]
    rows = {(e["name"], e["status"], e["detail"]) for e in claim["evidence_results"]}
    lines += [f"| {name} | {status} | {detail.replace('|', '/')} |"
              for name, status, detail in sorted(rows)]
    return lines + [""]


def verify_packet(output_dir, current_identity: Optional[Dict[str, Any]] = None):
    """Verify artifact bytes, evidence consistency, completeness and identity.

    The manifest is an unsigned local integrity record, not an attestation
    against an actor who can rewrite both the evidence and the manifest.
    """
    root = Path(output_dir)
    try:
        manifest = load_json_object(root / "manifest.json")
        artifacts = manifest.get("artifacts", {})
        valid = (manifest.get("schema_version") == 1 and isinstance(artifacts, dict)
                 and REQUIRED_PACKET_ARTIFACTS.issubset(artifacts)
                 and not ({"manifest.json", "human_review.json"} & set(artifacts)))
        if valid:
            valid = all(_artifact_path(root, name).is_file()
                        and file_digest(_artifact_path(root, name)) == sha
                        for name, sha in artifacts.items())
        checks = [evidence("artifact_integrity", "PASS" if valid else "FAIL")]
        if not valid:
            return checks
        saved = load_json_object(root / "evidence.json")
        config = load_json_object(root / "config.json")
        registry = validate_registry(load_json_object(root / "registry.json"))
        identity = manifest["identity"]
        consistent_identity = (
            saved.get("identity") == identity
            and digest(config) == identity["config_sha256"]
            and digest(registry) == identity["registry_sha256"]
            and digest({key: value for key, value in identity.items()
                        if key != "identity_sha256"}) == identity["identity_sha256"]
        )
        checks.append(evidence(
            "saved_identity_consistency", "PASS" if consistent_identity else "FAIL"))
        expected = [_run_key(*key) for key in manifest["expected_runs"]]
        runs = saved["runs"]
        keys = [_run_key(run["cell"], run["seed"]) for run in runs]
        missing = [list(key) for key in expected if key not in keys]
        completed = [_run_key(*key) for key in manifest["completed_runs"]]
        complete_schema = (bool(expected) and len(set(expected)) == len(expected)
                           and len(set(keys)) == len(keys)
                           and set(keys).issubset(expected)
                           and len(completed) == len(keys)
                           and set(completed) == set(keys)
                           and manifest["missing_runs"] == missing
                           and digest(expected) == identity["expected_runs_sha256"])
        checks.append(evidence("run_accounting", "FAIL" if not complete_schema
                               else "NOT_RUN" if missing else "PASS", actual=missing))
        experiment_checks = [normalize_check(check)
                             for check in saved["experiment_checks"]]
        actual_runs = {key: run for key, run in zip(keys, runs)}
        computed_claims = []
        required_artifacts = set()
        all_checks = list(experiment_checks)
        for check in experiment_checks:
            required_artifacts.update(check["artifacts"])
        for run in runs:
            run["checks"] = [normalize_check(check) for check in run["checks"]]
            all_checks.extend(run["checks"])
            required_artifacts.update(run["artifacts"])
            for check in run["checks"]:
                required_artifacts.update(check["artifacts"])
            required_artifacts.update(
                case["artifact"] for case in run["cases"] if case.get("artifact"))
        checks.append(evidence(
            "referenced_artifacts",
            "PASS" if required_artifacts.issubset(artifacts) else "FAIL"))
        for claim in registry["claims"]:
            sources = (
                [experiment_checks] if claim.get("scope", "run") == "experiment"
                else [actual_runs.get(key, {}).get("checks", []) for key in expected]
            )
            evidence_checks = []
            for source in sources:
                by_name = {check["name"]: check for check in source}
                for name in claim["evidence"]:
                    item = dict(by_name.get(name, evidence(name, "NOT_RUN")))
                    item["required"] = True
                    evidence_checks.append(item)
            computed_claims.append(evidence(
                claim["id"], aggregate_status(evidence_checks)))
        recorded_claims = {claim["id"]: claim["status"] for claim in saved["claims"]}
        claim_consistency = (
            len(recorded_claims) == len(computed_claims)
            and all(recorded_claims.get(claim["name"]) == claim["status"]
                    for claim in computed_claims))
        computed_status = aggregate_status(computed_claims + all_checks)
        if missing and computed_status != "FAIL":
            computed_status = "NOT_RUN"
        checks.append(evidence(
            "evidence_consistency", "PASS" if claim_consistency
            and manifest["machine_status"] == computed_status else "FAIL"))
        checks.append(evidence("declared_machine_evidence", computed_status))
        if current_identity is not None:
            checks.append(evidence(
                "identity_match",
                "PASS" if identity == current_identity else "INCONCLUSIVE"))
        return checks
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            UnicodeError) as error:
        return [evidence(
            "artifact_integrity", "FAIL", detail=f"Invalid packet: {error}")]


def _artifact_path(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact paths must be relative within the packet")
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes packet directory")
    return result


def write_certificate(path, identity, checks, artifacts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [normalize_check(c) for c in checks]
    data = {"identity": identity, "checks": normalized,
            "artifacts": {a: file_digest(_artifact_path(path.parent, a))
                          for a in artifacts}}
    path.write_text(json.dumps(data, indent=2, allow_nan=False))
    return path
