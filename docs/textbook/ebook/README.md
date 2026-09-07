# Kindle reading edition

[Download the project textbook as an EPUB](healthcare-simulation-sdk-textbook.epub).

To put it on your Kindle:

1. Open [Amazon Send to Kindle](https://www.amazon.com/sendtokindle) and sign in.
2. Upload `healthcare-simulation-sdk-textbook.epub` and send it to your library.
3. Sync your Kindle using the same Amazon account.

Amazon accepts EPUB through Send to Kindle; its web service supports files up
to 200 MB. This book is approximately 1.2 MB. See
[Amazon’s transfer guidance](https://digprjsurvey.amazon.com/csad/help/node/TCUBEdEkbIhK07ysFu).
The EPUB was created locally; it has not been uploaded to your account.

## What is included

All sixteen chapters are retained, along with the glossary, bibliography,
worked exercises, peer-review questions, 47 equations and 101 cited code
excerpts. The linked table of contents and chapter files support navigation.
Text reflows with the reader’s font settings.

The interactive labs are static worked examples showing their default inputs,
charts, numerical results and assumptions. Tables become labeled records for
narrow screens. Exercise answers are expanded, and code excerpts follow the
main text as source-note appendices. Exact LaTeX strings remain in a notation
appendix. Local repository links become printed references; external scholarly
links remain available when the reader supports them and has network access.
The cover, diagrams, equations and charts are embedded images.

Instructions elsewhere in the manuscript about sliders, browser search and
review-note storage refer to the [original HTML edition](../index.html).
Those controls do not run inside the e-book. No reviewer’s personal browser
notes are exported into the EPUB.

The scientific status remains a reviewable author draft. Conversion to an
e-book does not change the validation scope or establish clinical evidence.

## Build and inspect

Run from the repository root using the existing dependency environment:

```bash
node docs/textbook/ebook/export_assets.cjs
.venv/bin/python docs/textbook/ebook/build_epub.py
.venv/bin/python docs/textbook/ebook/qa/check_epub.py docs/textbook/ebook/healthcare-simulation-sdk-textbook.epub
```

The visual exporter uses Playwright and an isolated Chrome profile. Use
`NODE_PATH` or `PLAYWRIGHT_MODULE` for an existing Playwright installation and
`CHROME_BIN` for the Chrome executable when needed. Check the exporter source
for its current environment defaults. The EPUB builder needs Pandoc, Python
BeautifulSoup and Pillow. The original manuscript remains unchanged.

`assets_manifest.json` binds the static figures and their data to the source
HTML. `edition_manifest.json` records the final EPUB identity and content
counts. [QA records](qa/quality_report.md) describe structural, content and
reading checks, with their actual scope. Full EPUBCheck validation and actual
Amazon conversion/Kindle hardware rendering are not claimed by local browser
previews.

`reading_copy.html` is the static prepackaging source. It is useful for
inspection, but the final reading checks examine the XHTML and images extracted
from the EPUB itself.
