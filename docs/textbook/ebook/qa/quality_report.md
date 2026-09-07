# Kindle reading edition: quality record

Date: 2026-09-06. [EPUB](../healthcare-simulation-sdk-textbook.epub) ·
[Transfer and build instructions](../README.md).

## Conversion and content

The reflowable EPUB contains all sixteen main chapters, 47 rendered equations,
eight static lab examples with default inputs/results, 101 cited source excerpts,
worked exercises and answers, peer-review questions, glossary and bibliography.
It includes a typographic cover and linked table of contents. Detailed source
notes and literal mathematical notation are retained in appendices.

Wide tables were converted to labeled records; interactive disclosures were
expanded; local file links became printed references. The browser-only fallback
notices were replaced by static labs. The source HTML and simulation code were
not changed. No personal browser review notes were included.

## Independent packaging and preservation checks

**131 checks passed**, with the full record in
[structure_results.json](structure_results.json) and the
[independent review](independent_review.md).

The checks inspect ZIP packaging, MIME type, XML, package metadata, manifest,
spine, navigation, resource/fragment resolution, image decoding and content
preservation. Every main chapter has an ordered table-of-contents target.
There are 38 content entries in the TOC and 39 text documents including the
cover. All 238 local references/fragments resolve; 113 external references are
classified separately.

The audit compared 1,665 source text units after explicitly disclosed markup
and printed-reference normalization. All 101 code-excerpt instances match
exactly, including duplicate instances when cited in several chapters. All 47
literal LaTeX expressions survive in the notation appendix. Exported visuals
used by the book are packaged byte-for-byte; 59 embedded PNGs including the
cover decode successfully and carry alternative text.

The first build placed headings inside containers that prevented Pandoc from
constructing the complete chapter navigation. This was corrected by promoting
chapter headings and identifiers before packaging. The final audit explicitly
checks all sixteen chapter targets and the split reading sequence.

## Reading inspection

**Eight reading checks passed** on the final EPUB. The browser preview examined all 40 spine documents at 600px and 360px, with no page-level horizontal overflow and all 58 HTML images loaded. Cover rendering was inspected separately. The final compact chart labels, dense equations, malnutrition text and wrapped source excerpts were visually reviewed. See [reading_review.md](reading_review.md), [reading_results.json](reading_results.json) and the six screenshots in `render/`. These checks use extracted EPUB XHTML/CSS/images, not the prepackaging HTML, and do not simulate actual Kindle pagination.

The first narrow-screen preview identified small chart labels and long source-symbol headings. Compact 360px chart captures and explicit text wrapping corrected those issues. The capacity lab’s flattened table text was also removed from its prose while its full patient records remained intact.

## Scope limits

This is a locally built EPUB prepared for Amazon Send to Kindle. No file was
uploaded to an Amazon account, and actual Amazon conversion, delivery and
physical Kindle rendering have not been exercised. Full EPUBCheck is also
**NOT_RUN** because its Java runtime/validator is unavailable locally. These
bounded independent checks are not a claim of complete EPUB conformance or
Kindle-device certification. [Primary guidance and validator availability](expectations.md)
records the relevant Amazon/W3C sources.

The original textbook’s scientific status remains a reviewable author draft.
An e-book conversion adds no clinical validation, external scientific peer
review or new production regression evidence. The labs are fixed teaching
examples, not live simulations.

## Identities

- EPUB SHA-256: `7bc2bde0666deb93438a6ad180c372f593fa016914268eed9bb3c8f4ad857461`
- Source HTML SHA-256: `ec23afc806913546c6d191145cc982d882102d1038e950d06ddeef0083e5c901`
- Original manuscript identity: `8305538b984aab9841c7c4fc2712e8f5245f8524cd832953426121798778ce21`
- File size: 1,186,487 bytes.
- Packager: pandoc 3.8.3; no build warnings.

See [edition_manifest.json](../edition_manifest.json) for build identities and
[assets_manifest.json](../assets_manifest.json) for original figure values,
dimensions and hashes. The reproducible scripts are supplied with this edition.
