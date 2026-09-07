# Textbook quality and review record

Edition: 2026-09-05. Status: **reviewable author draft**. External scientific
peer review and institution-specific clinical validation remain pending.

[Open the textbook](../index.html) · [Build and review instructions](../README.md)

## Delivered scope

- 16 chapters, approximately 21,800 chapter-body words, covering probability,
  hazards, stochastic dynamics, counterfactuals, Monte Carlo uncertainty, model
  metrics, controlled scores, decisions, data semantics, ITS evaluation,
  monitoring, malnutrition, the scenario atlas, software and verification.
- 8 offline interactive teaching labs, 47 vector equations with copyable
  LaTeX sources, 39 answered exercises and 52 chapter review questions.
- 36 primary/official bibliography entries and 101 source references resolving
  to 50 source files. Long excerpts disclose truncation and their full span.
- Searchable chapters, role-specific reading paths, responsive navigation,
  print styling and local review notes with explicit save/export behavior.
- A source manifest, editable chapter files, reproducible build and QA scripts.

## Final verification

| Check | Recorded result | Scope |
|---|---|---|
| Independent numerical comparisons | **137 PASS across 66 JavaScript calls** | Actual lab math checked against independent Python/SciPy/NumPy calculations and selected SDK helpers. |
| Chromium artifact checks | **30 PASS**, Chrome 152.0.7977.76 | Actual final HTML, including every range endpoint, search, review author/identity retention, export, print-mode disclosures, mobile navigation and no-JavaScript reading. |
| Offline asset loading | **PASS** | Browser recorded no HTTP(S) asset requests; all scripts, styles, equations and visual content are embedded. Following external references is separate. |
| Reflow | **PASS** at 320, 390, 768, 1024 and 1440 CSS pixels | No page-level horizontal overflow. Wide equations, source blocks and tables may scroll within their own containers. |
| Build integrity | **PASS** | All cited code symbols resolve; no duplicate static HTML/SVG IDs or broken internal chapter anchors. |
| Source/app/artifact applicability | **PASS** | Final HTML hash matches browser evidence; actual JavaScript hash matches numerical evidence; cited current files match the manifest. |
| Same-environment rebuild | **PASS** | Rebuilding the final sources produced byte-identical HTML. This does not establish byte identity across different dependency/font environments. |
| Python syntax and whitespace | **PASS** | Documentation builder/checker compile; Git diff whitespace check passes. |

[Full numerical results](numerical_results.json) and [full browser results](browser_results.json)
retain individual checks. The browser suite initially exposed long source-symbol
labels overflowing narrow screens. Wrapping those labels fixed the issue; the
final reflow checks passed. The numerical comparisons found no defect in the
covered teaching functions. These are documentation checks, not a newly run
production regression suite.

## Internal content review

Separate agents reviewed chapters they had not authored, against current code
and retained evidence. Their corrections include malnutrition's actual primary
endpoint and branch-specific denominator, horizon-bounded case descriptions,
fixed nurse-position label alignment, conditional model performance, capacity
scope and numerical-source rendering. This is internal agent cross-review;
it is not independent human scientific peer review.

- [Foundations and malnutrition review](review_foundations_case.md)
- [Prediction and scenario atlas review](review_prediction_atlas.md)
- [Inference and engine review](review_inference_engine.md)
- [Artifact and review-provenance review](review_artifact.md)
- [Independent lab numerical review](review_numerics.md)

The review-note audit found two provenance defects during development: export
implicitly renewed old manuscript identities, and reviewer attribution was
stored globally. Final code requires an explicit Save to renew a note, preserves
old identities on export, binds authors per chapter, and requires a name for a
reviewed status. Browser checks verify those behaviors against the final file.
Names remain self-reported; hashes and local notes are not digital signatures.

## Visual and accessibility inspection

Representative final screenshots were inspected for typography, readable
labels, chart units, hierarchy and clipping: [desktop](desktop.png),
[mobile](mobile.png), [Bayes lab](bayes_lab.png), [mobile lab](mobile_lab.png),
[ITS lab](its_lab.png), [equation](equation.png) and [malnutrition](malnutrition.png).
The mobile cover diagram uses a stacked layout. Equations use vector rendering
and retain readable sizing inside scrollable containers. Chart results also
appear as text, and every form control has an associated label.

Navigation and disclosures use native browser semantics. Offscreen mobile
navigation is inert; Escape/close restores focus in the tested interactions.
Selected primary text/background colors were checked for contrast (body text,
muted text, links, navigation labels and buttons exceed 4.5:1). Chart color is
supplemented by text labels. These checks do **not** establish WCAG conformance:
full screen-reader reading order, semantic MathML, assistive-technology behavior,
all keyboard paths, other browsers and actual printed/PDF pagination were not
formally audited. Print-mode styling and disclosure expansion/restoration were
tested; no separately reviewed PDF is delivered.

## Scientific limits

The labs are declared hypothetical teaching calculators, not full production
scenario runs. The fixed noise fixtures used in ITS and monitoring do not
estimate stochastic coverage, power or false-alarm rates. Numerical correctness
of a formula does not establish its scientific applicability.

The book documents meaningful implementation limitations rather than silently
correcting production behavior: score/label-conditioning scope, AP tie handling,
RNG draw-alignment requirements, snapshot inherited history, occupancy versus
incidence, and simplified paper-inspired workflows. It does not certify every
scenario or establish that published study designs are fully replicated.

The malnutrition chapter retains the actual scope of the additional checks:
a public synthetic three-cell/three-seed demonstration and stronger mechanism
verification, with imprecise mean evidence and incomplete subgroup/follow-up
information. It does not renew all original efficacy, LOS, readmission,
financial or equity claims. Clinical assumptions, local model validation and
the intended real-world study design still require qualified review.

## Exact artifact identities

- Baseline Git commit: `648c56ce330981449435a10201f2116f2f111c5c`
- Manuscript identity: `8305538b984aab9841c7c4fc2712e8f5245f8524cd832953426121798778ce21`
- Final HTML SHA-256: `ec23afc806913546c6d191145cc982d882102d1038e950d06ddeef0083e5c901`
- Checked JavaScript SHA-256: `34dd1a92226a2c23ae44ad2a43381561f4472411a14388c7adf85809cd098b26`

The manuscript identity includes the working-tree source hashes and build
inputs; it is not equivalent to the baseline revision. See the
[manifest](../source_manifest.json) for dependency versions, individual code
references and recorded hashes of dated local illustration artifacts.

Production code was not changed for this textbook task. The existing
implementation changes remain in the working tree. The repository README now
links to the book. All deliverables are saved locally; no site was published.
