#!/usr/bin/env python3
"""Build a portable, offline textbook from reviewed chapter JSON and local source.

Run from any directory with a Python environment containing mistune,
matplotlib and beautifulsoup4. No network requests or simulation runs occur.
"""
from __future__ import annotations

import ast
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sdk-book-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "sdk-book-cache"))
import matplotlib
matplotlib.use("Agg")
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
import mistune
import bs4
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ESC = html.escape
SCHEMA = 1
DATE = "2026-09-05"
PARTS = {1: "Foundations", 2: "Foundations", 3: "Foundations", 4: "Foundations", 5: "Foundations",
         6: "Prediction and decisions", 7: "Prediction and decisions", 8: "Prediction and decisions",
         9: "Inference and evaluation", 10: "Inference and evaluation", 11: "Inference and evaluation",
         12: "Applications", 13: "Applications", 14: "Implementation and assurance",
         15: "Implementation and assurance", 16: "Implementation and assurance"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_ref(ref: dict) -> dict:
    path = ROOT / ref["path"]
    if not path.is_file() or not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Missing/outside source: {path}")
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    start, end = 1, len(lines)
    if ref.get("symbol"):
        found = []

        def walk(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + node.name
                    if name == ref["symbol"]:
                        found.append(node)
                    walk(node.body, name + ".")

        walk(ast.parse(raw).body)
        if len(found) != 1:
            raise ValueError(f"Expected one symbol {ref['symbol']} in {path}; found {len(found)}")
        node = found[0]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = node.end_lineno
    excerpt_end = min(end, start + 69)
    text = "\n".join(f"{i:4d}  {lines[i-1]}" for i in range(start, excerpt_end + 1))
    return {**ref, "sha256": sha(raw), "line_start": start, "line_end": end,
            "excerpt_end": excerpt_end, "excerpt": text}


class Renderer(mistune.HTMLRenderer):
    def __init__(self, chapter):
        super().__init__(escape=False)
        self.chapter = chapter
        self.equations = 0
        self.headings = 0

    def heading(self, text, level, **attrs):
        self.headings += 1
        level = min(6, max(3, level + 1))
        return f'<h{level} id="{self.chapter}-section-{self.headings}">{text}</h{level}>\n'

    def block_code(self, code, info=None):
        if info != "math":
            return super().block_code(code, info)
        self.equations += 1
        expression = code.strip()
        buf = io.BytesIO()
        math_to_image("$" + expression + "$", buf, format="svg", dpi=144,
                      prop=FontProperties(size=14), color="#172a30")
        markup = buf.getvalue().decode("utf-8")
        markup = markup[markup.index("<svg"):]
        prefix = f"eq-{self.chapter}-{self.equations}-"
        for ident in re.findall(r'id="([^"]+)"', markup):
            markup = markup.replace(f'id="{ident}"', f'id="{prefix}{ident}"')
            markup = markup.replace(f'#{ident}"', f'#{prefix}{ident}"')
            markup = markup.replace(f'#{ident})', f'#{prefix}{ident})')
        markup = re.sub(r'<metadata>.*?</metadata>', '', markup, flags=re.S)
        markup = markup.replace('<svg ', '<svg role="img" aria-label="' + ESC(expression, quote=True) + '" ', 1)
        label = f"{int(self.chapter[:2])}.{self.equations}"
        # The exact source is visible and copyable in a disclosure for assistive readers.
        return (f'<figure class="equation" id="equation-{label}">{markup}'
                f'<figcaption>Equation {label}</figcaption><details class="equation-source">'
                f'<summary>Equation in LaTeX / plain-text notation</summary><pre><code>{ESC(expression)}</code></pre>'
                '</details></figure>\n')


def markdown(text, renderer):
    return mistune.create_markdown(renderer=renderer, plugins=["table", "strikethrough"])(text)


GLOSSARY = [
    ("Actionability", "Whether a prediction can lead to a feasible and timely beneficial action."),
    ("AP", "Average precision: a precision–recall summary whose exact convention, including ties, must be specified."),
    ("AR(1)", "A process whose current value depends linearly on its previous value and an innovation."),
    ("AUC", "Probability a randomly selected positive ranks above a negative, with half credit for tied scores."),
    ("Bias", "Expected estimator minus its target under a specified data-generating mechanism."),
    ("Calibration", "Agreement between stated probabilities and event frequencies for a defined population and outcome."),
    ("Censoring", "Incomplete observation of an outcome or follow-up interval; not the same as observing no event."),
    ("Common random numbers", "A coupling of simulation inputs designed to improve precision of comparisons; its benefit depends on covariance."),
    ("Counterfactual", "Outcome under a specified alternative policy or history."),
    ("Coverage", "Probability that an interval procedure contains the target over repeated datasets."),
    ("CUSUM", "Cumulative sum detector that accumulates deviations relative to a reference and signals at a threshold."),
    ("Discrimination", "Ability of a score to rank or separate cases and noncases."),
    ("Estimand", "Precisely defined target quantity: population, policy contrast, outcome, horizon and aggregation."),
    ("Estimator", "Procedure mapping observed data into an estimate of a target."),
    ("HAC", "Heteroskedasticity- and autocorrelation-consistent covariance estimation, subject to its regularity and finite-sample limits."),
    ("Hazard", "Instantaneous event rate conditional on remaining event-free; measured in inverse time."),
    ("Identification", "Assumptions under which the desired causal quantity can be expressed using observed-data information."),
    ("Incidence", "New events over a defined interval and population at risk."),
    ("Interference", "One entity’s outcome depends on actions or resources allocated to other entities."),
    ("ITS", "Interrupted time series: models an outcome trajectory across an intervention break with explicit trend assumptions."),
    ("Label maturity", "Time when the full outcome window required to evaluate a prediction has become observable."),
    ("MCSE", "Monte Carlo standard error: numerical uncertainty due to a finite number of independent replications."),
    ("Occupancy", "Presence in a state at an observation time; sums may represent person-time rather than incident cases."),
    ("Oracle", "Information or target known inside a configured simulator but usually unavailable in real-world observation."),
    ("Paired replication", "Two policy outcomes generated within one coupled simulated world; independent pairs support inference."),
    ("PPV", "Positive predictive value: proportion of flagged observations that are true cases at the specified horizon."),
    ("Provenance", "Recorded source, inputs, model, environment and execution lineage; a hash detects changes but does not establish truth."),
    ("Seed spread", "Variation among individual simulation runs under fixed inputs."),
    ("Sensitivity", "True-positive rate: probability of a positive flag conditional on an actual case."),
    ("Specificity", "True-negative rate: probability of a negative flag conditional on an actual noncase."),
    ("Transportability", "Whether a conclusion applies to another population, setting, workflow or period under defensible assumptions."),
    ("Type I error", "Probability of rejecting a true null hypothesis under the specified analysis procedure."),
]

HERO_SVG = '''<svg viewBox="0 0 820 208" role="img" aria-label="A modeled population enters two policy worlds, which produce outcomes; observable data support analysis, while their paired contrast defines simulation truth.">
<defs><marker id="hero-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#55776e"/></marker></defs>
<g font-family="system-ui,sans-serif" font-size="13" fill="#172a30">
<rect x="0" y="57" width="132" height="78" rx="5" fill="#e2ebe5"/><text x="15" y="84" font-weight="600">Population</text><text x="15" y="104">Risks · histories</text><text x="15" y="122">Assumptions θ</text>
<rect x="182" y="6" width="280" height="76" rx="5" fill="#dcece6"/><text x="197" y="32" font-weight="600">Factual policy</text><text x="197" y="53">Evolve → score → allocate → measure</text><text x="197" y="71" font-size="11">Usual care + configured intervention</text>
<rect x="182" y="112" width="280" height="76" rx="5" fill="#eee5d6"/><text x="197" y="139" font-weight="600">Comparator policy</text><text x="197" y="160">Evolve → usual care → measure</text><text x="197" y="178" font-size="11">Separate state · coupled random inputs</text>
<rect x="512" y="6" width="122" height="76" rx="5" fill="#e2ebe5"/><text x="527" y="33" font-weight="600">Factual loss</text><text x="527" y="56">L₁</text>
<rect x="512" y="112" width="122" height="76" rx="5" fill="#eee5d6"/><text x="527" y="139" font-weight="600">Baseline loss</text><text x="527" y="162">L₀</text>
<text x="666" y="66" font-weight="600">Observed analysis</text><text x="666" y="86" font-size="12">Uses available data</text><text x="666" y="146" font-weight="600">Oracle contrast</text><text x="666" y="166" font-size="12">Conditional on θ</text>
</g><g fill="none" stroke="#55776e" stroke-width="1.7" marker-end="url(#hero-arrow)"><path d="M132 83 L160 83 L160 44 L182 44"/><path d="M132 109 L160 109 L160 150 L182 150"/><path d="M462 44 L512 44"/><path d="M462 150 L512 150"/><path d="M634 44 L650 44 L650 64 L660 64"/><path d="M634 150 L660 150"/><path d="M573 82 L573 98 L650 98 L650 136 L660 136"/></g></svg>'''
HERO_SVG += '''<div class="hero-mobile-flow" role="img" aria-label="A modeled population branches into factual and comparator policies. Factual data support observed analysis; paired outcomes define conditional oracle truth."><div class="flow-wide"><strong>Population</strong>Risks · histories · assumptions</div><div class="flow-arrow">↓ Two policy worlds ↓</div><div><strong>Factual policy</strong>Evolve → score → allocate → measure<br><b>Loss L₁</b></div><div class="flow-comparator"><strong>Comparator</strong>Evolve → usual care → measure<br><b>Loss L₀</b></div><div class="flow-wide"><strong>Factual data → observed analysis</strong>Requires an appropriate study design</div><div class="flow-wide flow-comparator"><strong>Paired outcomes → oracle contrast</strong>Conditional on the modeled assumptions</div></div>'''


def main():
    chapter_files = sorted((HERE / "chapters").glob("*.json"))
    chapters = [json.loads(p.read_text()) for p in chapter_files]
    if [c["number"] for c in chapters] != list(range(1, 17)):
        raise ValueError("Expected exactly sixteen ordered chapters")
    ids = {c["id"] for c in chapters}
    for c in chapters:
        for key in ("id", "title", "subtitle", "objectives", "body", "code_refs", "references", "review_questions", "exercises"):
            if not c.get(key):
                if key not in ("references",):
                    raise ValueError(f"Missing {key}: {c['id']}")
        if not set(c["prerequisites"]).issubset(ids):
            raise ValueError(f"Invalid prerequisite: {c['id']}")
    title_by_id = {c["id"]: c["title"] for c in chapters}
    code = {c["id"]: [source_ref(r) for r in c["code_refs"]] for c in chapters}
    sources = {r["path"]: r["sha256"] for refs in code.values() for r in refs}
    artifacts = {}
    for c in chapters:
        for r in c.get("artifact_refs", []):
            p = ROOT / r["path"]
            if p.exists() and sha(p.read_bytes()) != r["sha256"]:
                raise ValueError(f"Dated evidence changed; review the chapter before updating its recorded hash: {p}")
            artifacts[r["path"]] = {"sha256": r["sha256"], "claim": r["claim"]}
    inputs = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in chapter_files + [HERE / "assets/style.css", HERE / "assets/app.js", Path(__file__)]}
    baseline = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dependencies = {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
                    "mistune": mistune.__version__, "beautifulsoup4": bs4.__version__}
    identity_payload = {"schema": SCHEMA, "baseline": baseline, "sources": sources,
                        "dated_artifacts": artifacts, "build_dependencies": dependencies,
                        "manuscript_inputs": inputs}
    identity = sha(json.dumps(identity_payload, sort_keys=True).encode())
    words = sum(len(re.findall(r"\b[\w’'-]+\b", c["body"])) for c in chapters)
    bibliography = {}
    equation_count = 0
    rendered = []
    for c in chapters:
        renderer = Renderer(c["id"])
        body = markdown(c["body"], renderer)
        equation_count += renderer.equations
        prereqs = " ".join(f'<a href="#{p}">{int(p[:2]):02d} · {ESC(title_by_id[p])}</a>' for p in c["prerequisites"])
        refs = []
        for r in code[c["id"]]:
            label = r.get("symbol") or r["path"]
            truncated = r["excerpt_end"] < r["line_end"]
            stamp = f"Lines {r['line_start']}–{r['excerpt_end']}" + (f" of {r['line_start']}–{r['line_end']}; selected opening excerpt" if truncated else "; full referenced definition" if r.get("symbol") else "; full file")
            refs.append(f'<details class="code-reference"><summary>{ESC(label)}</summary><div class="details-body"><p>{ESC(r["claim"])}</p><p><a href="../../{ESC(r["path"])}">{ESC(r["path"])}</a></p><p class="source-stamp">{stamp}<br>File SHA-256: {r["sha256"]}</p><pre><code>{ESC(r["excerpt"])}</code></pre></div></details>')
        bibliography_rows = []
        for r in c["references"]:
            bibliography.setdefault(r["url"], {**r, "chapters": []})["chapters"].append(c["number"])
            bibliography_rows.append(f'<li><a href="{ESC(r["url"], quote=True)}">{ESC(r["title"])}</a>. {ESC(r.get("note", ""))}</li>')
        exercises = ''.join(f'<details><summary>Exercise {c["number"]}.{i+1} · {ESC(e["question"])}</summary><div class="details-body">{markdown(e["answer"], renderer)}</div></details>' for i,e in enumerate(c["exercises"]))
        rendered.append(f'''<section class="chapter" id="{c['id']}" aria-labelledby="title-{c['id']}">
<header class="chapter-head"><div class="chapter-kicker">{ESC(PARTS[c['number']])} · Chapter {c['number']:02d}</div><h2 id="title-{c['id']}">{ESC(c['title'])}</h2><p class="subtitle">{ESC(c['subtitle'])}</p>
<div class="objectives"><strong>After this chapter, you can</strong><ul>{''.join('<li>'+ESC(o)+'</li>' for o in c['objectives'])}</ul></div>
{('<p class="prerequisites">Builds on: '+prereqs+'</p>') if prereqs else ''}</header>
<div class="chapter-body">{body}</div>
<h3>Practice and check your reasoning</h3><p class="mini-label">Open an exercise for its worked answer</p>{exercises}
<div class="chapter-review"><h3>Questions for peer review</h3><ol>{''.join('<li>'+ESC(q)+'</li>' for q in c['review_questions'])}</ol></div>
<h3>Trace the implementation</h3><p class="code-reference">These excerpts were read from the local source at build time. They support the specified behavior; they do not prove every path is correct. Expand to inspect numbered lines and the file identity.</p>{''.join(refs)}
{('<details><summary>Primary references and further study</summary><ol class="references">'+''.join(bibliography_rows)+'</ol></details>') if bibliography_rows else ''}
</section>''')
    nav = []
    part = None
    for c in chapters:
        if PARTS[c["number"]] != part:
            part = PARTS[c["number"]]
            nav.append(f'<div class="nav-part">{ESC(part)}</div>')
        nav.append(f'<a class="chapter-link" href="#{c["id"]}"><span class="num">{c["number"]:02d}</span><span>{ESC(c["title"])}</span></a>')
    glossary = ''.join(f'<div><dt>{ESC(k)}</dt><dd>{ESC(v)}</dd></div>' for k,v in GLOSSARY)
    references = ''.join(f'<li><a href="{ESC(url, quote=True)}">{ESC(r["title"])}</a>. {ESC(r.get("note", ""))} <span class="source-stamp">Chapters {", ".join(map(str, sorted(set(r["chapters"]))))}.</span></li>' for url,r in bibliography.items())
    source_rows = ''.join(f'<tr><td><a href="../../{ESC(p)}">{ESC(p)}</a></td><td class="source-stamp">{h}</td></tr>' for p,h in sorted(sources.items()))
    options = ''.join(f'<option value="{c["id"]}">{c["number"]:02d} · {ESC(c["title"])}</option>' for c in chapters)
    css = (HERE / "assets/style.css").read_text()
    js = (HERE / "assets/app.js").read_text()
    page = f'''<!doctype html><html lang="en" data-identity="{identity}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="An illustrated, source-backed textbook on probability, machine learning, causal evaluation and engineering in the Healthcare Intervention Simulation SDK."><title>The science of simulating healthcare AI · Healthcare Simulation SDK</title><style>{css}</style></head><body>
<a class="skip" href="#main">Skip to the book</a><aside class="sidebar" id="book-navigation"><a class="brand" href="#main"><small>Healthcare Simulation SDK</small>The science of<br>simulating healthcare AI</a><nav aria-label="Chapters">{''.join(nav)}<div class="nav-part">Reference desk</div><a class="chapter-link" href="#glossary">Notation &amp; glossary</a><a class="chapter-link" href="#bibliography">Bibliography</a><a class="chapter-link" href="#evidence">Source &amp; review record</a></nav><div class="sidebar-footer">Author draft · {DATE}<br>External peer review pending<br><a href="README.md">Build &amp; maintenance guide</a></div></aside>
<div class="reading-progress" aria-hidden="true"></div><div class="shell"><div class="topbar"><button class="menu-toggle" aria-controls="book-navigation" aria-expanded="false">Contents</button><span class="edition">A source-backed field textbook<br><strong>16 chapters · 8 interactive labs</strong></span><div class="search-wrap"><label class="sr-only" for="book-search">Search chapters</label><input type="search" id="book-search" placeholder="Search the book…" autocomplete="off" aria-controls="search-results"><div class="search-results" id="search-results" hidden></div></div><button class="print-button">Print / save PDF</button><button id="review-toggle" class="primary" aria-controls="review-panel" aria-expanded="false">Review notes</button></div>
<main class="book" id="main"><header class="hero"><div class="eyebrow">Probability · prediction · decisions · evidence</div><h1>The science of<br>simulating healthcare AI</h1><p class="deck">How this project turns assumptions about patients, models and workflows into experiments—and how to challenge the conclusions.</p><div class="hero-tags"><span>16 chapters</span><span>{words:,} chapter words</span><span>8 visual labs</span><span>Offline &amp; portable</span></div><figure class="hero-figure">{HERO_SVG}<figcaption>Two policy worlds answer a conditional simulation question. An analysis of observable data must earn its own evidential claim.</figcaption></figure><div class="book-status"><strong>Reviewable author draft · {DATE}.</strong> This manuscript combines derivations, worked examples, actual source excerpts and explicit limitations. Internal numerical and source review is documented below. External scientific peer review and clinical validation remain pending.</div></header>
<div class="path-grid"><div class="path-card"><strong>Clinical &amp; operations</strong>Begin with the decision, examine scarce capacity, then follow the malnutrition case and its evidence gaps.<a href="#01-orientation">01 → 08 → 12 → 16</a></div><div class="path-card"><strong>Quantitative &amp; research</strong>Build from probability and counterfactuals to Monte Carlo precision, data semantics and study design.<a href="#02-probability">02 → 03 → 04 → 05 → 09 → 10 → 11</a></div><div class="path-card"><strong>Engineering &amp; assurance</strong>Connect model behavior to the state contract, source identity and layered verification.<a href="#07-controlled-models">07 → 13 → 14 → 15 → 16</a></div></div>
<noscript><p class="no-js">All chapters, equations, source excerpts and exercise answers remain available without JavaScript. Interactive controls, search and local review-note storage require JavaScript. The surrounding chapters provide noninteractive worked examples.</p></noscript>
{''.join(rendered)}
<section class="chapter" id="glossary"><div class="chapter-kicker">Reference desk</div><h2>Notation and glossary</h2><p>Symbols are defined where used. The same letter may have different local meanings; always check the chapter’s definition and units.</p><div class="table-wrap"><table><thead><tr><th>Symbol</th><th>Usual meaning in this book</th><th>Unit or condition</th></tr></thead><tbody><tr><td>i, t, r</td><td>Entity, time index, replication index</td><td>Indexes, not observations assumed independent</td></tr><tr><td>p, π, Se, Sp</td><td>Risk, prevalence, sensitivity, specificity</td><td>Dimensionless probabilities at a declared horizon</td></tr><tr><td>λ, Δt</td><td>Hazard and time interval</td><td>Inverse time and matching time units</td></tr><tr><td>θ, U</td><td>Model assumptions and simulated randomness</td><td>Fixed inputs versus repeated worlds</td></tr><tr><td>π₀, π₁; L₀, L₁</td><td>Policies and their losses</td><td>π with policy subscripts is not prevalence</td></tr><tr><td>Δ, Dᵣ</td><td>Mean policy contrast and one paired difference</td><td>Outcome units; sign must be declared</td></tr><tr><td>R, K</td><td>Replication count or resource budget; capacity</td><td>Context-specific; not interchangeable</td></tr><tr><td>ρ, σ</td><td>Correlation/persistence and standard deviation</td><td>Dimensionless and outcome/process units</td></tr></tbody></table></div><dl class="glossary-grid">{glossary}</dl></section>
<section class="chapter" id="bibliography"><div class="chapter-kicker">Reference desk</div><h2>Primary references</h2><p>External sources support the named mathematical or methodological ideas. Source code establishes what this project currently implements. A citation to a method does not certify a scenario’s clinical inputs.</p><ol class="references">{references}</ol></section>
<section class="chapter" id="evidence"><div class="chapter-kicker">Reference desk</div><h2>Source identity and review record</h2><p>This edition describes a local working tree containing implementation changes. It is not identified completely by the baseline Git revision. The manuscript identity binds chapter inputs, visual code, build logic and each cited source file by content hash.</p><p class="source-stamp">Baseline commit: {baseline}<br>Manuscript identity: {identity}<br>Edition date: {DATE}</p><p><a href="source_manifest.json">Machine-readable source manifest</a> · <a href="qa/quality_report.md">Documentation quality report</a> · <a href="README.md">Build and review instructions</a></p><p>Internal agent cross-review records: <a href="qa/review_foundations_case.md">foundations and malnutrition</a>, <a href="qa/review_prediction_atlas.md">prediction and scenario atlas</a>, <a href="qa/review_inference_engine.md">inference and engine</a>. These are computational/editorial review records, not independent human peer-review decisions.</p><details><summary>Inspect all {len(sources)} cited source-file identities</summary><div class="table-wrap"><table><thead><tr><th>Repository-relative file</th><th>SHA-256 of full file</th></tr></thead><tbody>{source_rows}</tbody></table></div></details><p>External references require an internet connection when opened. Repository file links and QA records require the companion directory tree. The chapters, figures, equations, code excerpts and calculators are embedded in this HTML file and work offline.</p></section>
<footer class="footer">Healthcare Intervention Simulation SDK · educational technical manuscript · {DATE}. Reading this book or recording a note does not create clinical certification or a signed review decision. Rebuild after relevant source changes and reassess applicability of prior evidence.</footer></main></div>
<section class="review-panel" id="review-panel" role="region" aria-labelledby="review-title" hidden><h3 id="review-title">Chapter review notes</h3><p class="notice">Local browser notes with an exportable manuscript identity. Save before changing chapters. A reviewed status records your note; it does not certify clinical evidence.</p><label for="review-name">Reviewer name<input id="review-name" autocomplete="name"></label><label for="review-chapter">Chapter<select id="review-chapter">{options}</select></label><label for="review-status">Your review status<select id="review-status"><option value="pending">Pending</option><option value="questions">Questions to resolve</option><option value="changes_requested">Changes requested</option><option value="reviewed">Reviewed by named reviewer</option></select></label><label for="review-note">Evidence, corrections and remaining questions<textarea id="review-note" placeholder="Identify the claim, your evidence and the change needed."></textarea></label><p id="review-notice" class="notice" aria-live="polite"></p><div class="actions"><button id="review-save" class="primary">Save locally</button><button id="review-export">Export saved JSON</button><button id="review-close">Close</button></div></section>
<script>{js}</script></body></html>'''
    soup = BeautifulSoup(page, "html.parser")
    for table in soup.select(".chapter-body table"):
        table.wrap(soup.new_tag("div", attrs={"class": "table-wrap"}))
    for lab in soup.select("[data-lab]"):
        lab.append(soup.new_tag("p", attrs={"class": "no-js"}))
        lab.p.string = "Interactive teaching calculator. Enable JavaScript to vary inputs; the surrounding chapter gives the formula, assumptions and worked example."
    # Fail closed on unresolved internal links or duplicate IDs (including SVG definitions).
    all_ids = [tag["id"] for tag in soup.select("[id]")]
    if len(all_ids) != len(set(all_ids)):
        from collections import Counter
        raise ValueError(f"Duplicate IDs: {[i for i,n in Counter(all_ids).items() if n>1]}")
    broken = [a["href"] for a in soup.select('a[href^="#"]') if a["href"][1:] not in all_ids]
    if broken:
        raise ValueError(f"Broken anchors: {broken}")
    (HERE / "index.html").write_text(str(soup), encoding="utf-8")
    manifest = {**identity_payload, "manuscript_identity": identity, "edition_date": DATE,
                "status": "Author draft; external peer review pending", "chapter_word_count": words,
                "chapter_count": len(chapters), "equation_count": equation_count,
                "lab_count": len(soup.select('[data-lab]')), "bibliography_count": len(bibliography),
                "source_references": {k: [{a:b for a,b in r.items() if a != "excerpt"} for r in refs] for k,refs in code.items()},
                "html_sha256": sha((HERE / 'index.html').read_bytes())}
    (HERE / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k:manifest[k] for k in ['manuscript_identity','chapter_count','chapter_word_count','equation_count','lab_count','bibliography_count']}, indent=2))


if __name__ == "__main__":
    main()
