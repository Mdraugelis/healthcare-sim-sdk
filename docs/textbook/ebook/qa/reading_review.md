# EPUB reading preview

The final EPUB passes all eight automated reading checks at 600px and 360px viewport widths. Its SHA-256 is `7bc2bde0666deb93438a6ad180c372f593fa016914268eed9bb3c8f4ad857461`.

All sixteen main chapters appear in navigation. Packaged images load with JavaScript disabled, figures stay within the reading viewport, and no spine document has page-level horizontal overflow. No remote resources or browser runtime errors were observed.

The visual review covered the cover, dense Bayes equation, narrow Bayes and capacity labs, malnutrition prose, and source excerpts. Six screenshots are saved under [render](render/); the Bayes screenshot is grayscale. Chart labels now use plots redrawn at a native 360px width, producing readable labels when fitted to a narrow reader. Original wider chart assets remain available for comparison. The duplicated flattened capacity table was removed from results prose; structured patient records remain. Long source symbols and file hashes wrap after the stylesheet correction.

Evidence: [machine-readable results](reading_results.json), [reproducible preview script](check_reading.cjs), [grayscale Bayes lab](render/bayes-lab-360-grayscale.png), [capacity lab](render/capacity-lab-360.png), [source excerpt](render/source-notes-360.png).

This is a browser preview of the actual EPUB's extracted XHTML, CSS, and images. It does not validate Amazon conversion, Kindle pagination, physical hardware, or e-ink rendering. Screenshots show a scrolling viewport; content continuing below its lower edge is not a cropped figure in the EPUB.
