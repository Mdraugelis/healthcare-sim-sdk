# The Data Boundary

This SDK is open-sourceable, but the work done *with* it often is not.
Campus-level rates, benchmark figures under vendor confidentiality, staff
names, internal work-item numbers, and raw institutional actuals must
never enter tracked files or git history. This document describes the
boundary and the three mechanisms that keep material on the right side of
it.

The design goal is **default-private**: confidential material stays out of
version control unless someone *explicitly* promotes a sanitized copy.
That inverts the usual git default ("tracked unless ignored"), where every
new directory is a fresh chance to forget an ignore rule.

## The three-way split

| Layer | What lives here | Tracked? |
|---|---|---|
| **Generic / reusable** | Runner logic, config *structure*, `*.example.*` templates | ✅ tracked |
| **Institution values** | Campus configs, frozen model params, actuals, `outputs/`, ad-hoc analyses | 🔒 gitignored |
| **Promotion** | A sanitized template distilled from institution values | ✅ tracked, deliberately |

The split is *values vs. structure*, not *experiment vs. example*. The
shape of an experiment (its Hydra config keys, its runner) is generic and
belongs in the repo. Only the confidential *values* plugged into that
shape stay local.

## Rule 1 — Put confidential work in a private location

Two conventions, either works, both are gitignored automatically:

- **A `local/` directory** — anywhere in the tree, at any depth. Drop
  working material, scratch analyses, and institution configs here.
- **A `*.private.*` suffix** — e.g. `campus_gmc.private.yaml`,
  `actuals.private.md`. Use this when a private file needs to sit next to
  its tracked siblings.

Because these are matched by pattern, a brand-new subdirectory is covered
with **zero new ignore rules** — which is exactly the gap that caused the
past leak (`analyses/` was untracked but had no ignore rule, so a
`git add -A` would have committed it).

Scenario-specific ignores still apply too — e.g. the malnutrition scenario
keeps `configs/campus/*` private and tracks only `example.yaml`.

## Rule 2 — Promote by sanitizing into a `*.example.*` template

To share the *structure* of a private config, copy it to a `*.example.*`
file, replace every confidential value with a generic placeholder, and
commit the example. The `*.example.*` file is explicitly allowlisted in
`.gitignore` (`!...example.yaml`). Promotion is a deliberate act, not a
default.

```
configs/campus/site_a.private.yaml  # 🔒 real values, never tracked
configs/campus/example.yaml         # ✅ sanitized structure, tracked
```

## Rule 3 — The commit-time guard catches what the rules miss

Rules 1 and 2 depend on people remembering them, and a `.gitignore` rule
only stops an accidental `git add` — it does nothing if you hardcode a
campus name straight into tracked source. So there is a
convention-independent backstop: a pre-commit hook
(`scripts/check_no_institution_terms.py`) that scans **staged content and
the commit message** against a denylist of confidential terms and blocks
the commit on a match.

The denylist is intentionally split so the guard never leaks the very
terms it protects:

- **`.institution-denylist.example`** — tracked, safe, generic. Only
  placeholders and pattern examples.
- **`.institution-denylist`** — gitignored, local. Your real terms. Never
  committed.

### Setup (once per clone that handles confidential data)

```bash
cp .institution-denylist.example .institution-denylist
# edit .institution-denylist — add org names, campus codes, vendors, etc.
pre-commit install                         # staged-file scanning
pre-commit install --hook-type commit-msg  # commit-message scanning
```

If `.institution-denylist` is absent the hook prints a one-time notice and
**passes** — external contributors with no confidential data are never
blocked by it.

### Denylist format

One rule per line; blank lines and `#` comments ignored.

| Line | Meaning |
|---|---|
| `AcmeHealth` | case-insensitive literal substring |
| `re:\bSITE_A\b` | case-insensitive regex (use `\b` word boundaries to avoid false positives like `SITE_ABLE`) |

If the guard fires on a legitimate string, tighten the offending rule in
your local `.institution-denylist` — don't weaken the boundary globally.

## What this does *not* cover

- **Existing git history and GitHub pull-request refs.** The guard is
  forward-looking. Purging terms already committed requires a history
  rewrite (`git filter-repo`), and PR refs (`refs/pull/N/head`) can only
  be removed by GitHub Support.
- **Semantic leaks.** A confidential *number* with no denylisted term next
  to it will pass. The denylist catches named entities, not every
  sensitive value — Rules 1 and 2 remain the primary defense.

## Quick reference

```
local/**                 🔒 private working space (any depth)
*.private.*              🔒 private value files
*.example.*             ✅ sanitized, tracked templates
.institution-denylist   🔒 your real terms (local only)
.institution-denylist.example   ✅ safe template
```
