# Independent artifact review

Reviewed `docs/textbook/build.py`, `assets/app.js`, relevant styling, and the generated HTML on 2026-09-05. No unresolved blocking defect remained in the reviewed source after the author applied the corrections below. Browser layout and interaction testing is a separate QA step; this record does not claim formal accessibility conformance.

## Concrete findings and disposition

1. **Export renewed old review evidence without an explicit review action — corrected.** The original export handler called `persist()`. Opening an older-manuscript note and clicking Export silently changed its manuscript identity and timestamp while retaining its reviewed status. Export now serializes saved records without updating them. Explicit Save renews the selected record and records its previous manuscript identity.
2. **Reviewer attribution was global rather than attached to each note — corrected.** Changing the reviewer name for one chapter could make other chapters appear to belong to that reviewer. Schema 2 records a reviewer on each chapter, migrates the available legacy attribution before edits, and rejects the “reviewed” status without a nonblank name. These remain self-reported local notes, not authenticated signatures.
3. **JavaScript-disabled content was overstated — corrected.** The original notice promised an initial visible result in each lab, but the static HTML contained only explanatory placeholders. The notice now accurately points to the surrounding noninteractive worked examples. Chapters, SVG equations, source excerpts, and exercise disclosures are embedded; interactive calculations require JavaScript.
4. **Search silently omitted later matches — corrected.** Search originally returned only the first 12 matching sections with no indication of truncation. It now returns all matching chapter/reference sections.

The author also made the offscreen mobile navigation inert, returns focus when closing it with Escape, and removed a duplicate intersection observer. These source changes were inspected; their device and assistive-technology behavior still belongs to browser QA.

## Verification performed

- Executed the actual revised review-note code in an isolated Node harness. Confirmed legacy migration; export preserves an older record's identity and timestamp; separate Alice/Bob records retain separate authors; an unnamed reviewed record is not saved; and explicit renewal records both current and previous manuscript identities. All five checks passed.
- Inspected generated HTML: 47 equations have nonempty accessible labels and 47 visible, copyable LaTeX-source disclosures. All 969 static HTML/SVG IDs are unique, and SVG fragment/clip references resolve. The LaTeX source is escaped rather than interpreted as HTML. The source disclosure supports inspection and copying; it is not a substitute for semantic MathML or manual screen-reader testing.
- Inspected 101 source-reference records. The 47 truncated excerpts visibly identify their displayed line interval, complete referenced interval, and “selected opening excerpt” status. Full definitions/files receive separate labels. The hashes identify the full source files, not only the displayed excerpts; linked complete files require the companion repository tree.
- Confirmed the build resolves cited symbols and checks source paths remain inside the repository. It rejects missing symbols, duplicate static IDs, and broken internal anchors. The manifest records manuscript inputs, cited file identities, baseline revision, and generated HTML hash. The edition explicitly describes working-tree content rather than equating it to the baseline commit.
- Confirmed the generated page has no remote script, stylesheet, image, iframe, or form dependency. Source inspection found no fetch, XHR, WebSocket, or telemetry path. External bibliography links deliberately require network access when followed; repository/QA links deliberately require companion files.
- Review-note strings enter form values and exported JSON, and search text uses `textContent`; neither is injected as executable markup. Calculator HTML interpolates fixed authored text or controlled numeric inputs. The builder intentionally permits trusted chapter-authored raw HTML; it is not an untrusted-document sanitizer.
- Native inputs and disclosures have labels; calculator results use polite live regions; equation and chart SVGs have accessible names. Print handlers open disclosures and restore their prior state. Visual contrast, zoom/reflow, complete keyboard traversal, and screen-reader reading order require the separate browser review.

## Scope and maintenance

This audit changed only this QA record; the author corrected the build/application files. The generated artifact must be rebuilt after those source changes, then its current manifest and browser QA must be checked together. Local storage can be unavailable or cleared; the UI reports storage failures and exports explicitly saved in-memory records. Export is not a cryptographic signature, and a hash detects identity changes without establishing scientific truth or reviewer authenticity.
