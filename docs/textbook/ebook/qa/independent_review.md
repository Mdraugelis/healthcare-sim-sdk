# Independent internal review of the Kindle EPUB

Date: 2026-09-06. Reviewed artifact: `healthcare-simulation-sdk-textbook.epub`, SHA-256 `0859b29cd325e3cc8c10a21cf4ef20e993b9ed852d99e43fdd367cf451f74328`.

The final file passes **131 bounded structural and content checks** recorded in `structure_results.json`. The checker is independent of the EPUB builder and operates on the actual ZIP archive. It does not claim complete EPUB conformance or successful Amazon conversion.

## Results

- ZIP CRC, first-entry/uncompressed MIME declaration, container, XML, package metadata, manifest resources and spine references pass. The file is below Amazon's web-upload limit.
- All sixteen main chapter targets appear in the navigation document in the intended order. The rebuilt edition has a multi-document reading spine; a preliminary edition's heading/splitting problem was corrected by the builder before this retained result.
- All 238 local resource/fragment references resolve. The audit identifies 113 external references; this is a link-classification count, not a fresh availability check of every external website.
- All 1,665 source text units across sixteen chapters and three reference sections are found in the book. Separate checks cover each chapter's title, objectives, review questions, exercise questions and answers.
- All 101 cited code-excerpt instances are preserved as exact text, including repeated excerpts. All 47 literal LaTeX strings are preserved in the notation appendix.
- All 57 exported equation/workflow/lab visuals, plus the selected compact hero diagram, match their expected image bytes. All 59 raster resources, including the cover, decode successfully; image elements have nonempty alternatives.
- No scripts, forms, buttons, select controls or collapsed `details` elements remain necessary for reading.

Content comparison explicitly accounts for three disclosed transformations: eight web-only JavaScript fallback notices are replaced by eight static labs; converter-marked printed repository paths add annotations to existing labels; and markup whitespace around punctuation can change without changing textual tokens. These adjustments do not remove substantive paragraphs from comparison. Cited code is checked separately without token normalization.

## Reproduction and limits

```bash
.venv/bin/python docs/textbook/ebook/qa/check_epub.py docs/textbook/ebook/healthcare-simulation-sdk-textbook.epub
```

The result is specific to the recorded EPUB and source-HTML hashes. No production simulation or broader software test suite was rerun.

**EPUBCheck remains NOT_RUN:** limited local searches found no available validator or working Java runtime. **Amazon conversion, account delivery and physical Kindle rendering remain NOT_RUN:** this work did not upload or send the book. Image decoding and content retention do not establish small-screen legibility; the builder's separate visual inspection should assess equations, narrow tables, code and pagination. This is internal computational review, not external scientific peer review.

Primary guidance and the local validator search are documented in `expectations.md`.
