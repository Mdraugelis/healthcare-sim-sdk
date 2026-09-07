# Kindle EPUB validation expectations

Research checked 2026-09-06. No document was sent to an Amazon account.

Amazon lists EPUB as a supported Send to Kindle input and recommends its web service for files up to 200 MB. The delivery service still needs to process the uploaded book; creating an EPUB locally does not demonstrate successful delivery or rendering on a particular Kindle. The official help is readable at [Amazon's document-transfer guidance](https://digprjsurvey.amazon.com/csad/help/node/TCUBEdEkbIhK07ysFu). The canonical [Send to Kindle page](https://www.amazon.com/sendtokindle) returned HTTP 503 during this research, so the delivery UI itself was not inspected.

The structural audit follows these bounded requirements from [EPUB 3.3](https://www.w3.org/TR/epub-33/): the ZIP begins with an uncompressed `mimetype` containing exactly `application/epub+zip`; `META-INF/container.xml` locates the package; package metadata identifies the book; the manifest declares resources; the spine supplies reading order; and an XHTML navigation document supplies the table of contents. Resource and fragment references must resolve. Content documents reached through navigation belong in the spine. These checks cover important requirements without replacing the complete specification.

For this particular adaptation, additionally verify all sixteen chapters, worked examples, questions, solutions, citations and code excerpts survive; equations and diagrams have readable packaged images and alternatives; interactive calculators have informative static replacements; local computer paths do not remain broken links; and JavaScript/form controls are not required to read the book. Inspect the treatment of wide tables and code in a narrow viewport. These are project acceptance expectations, not a claim that every listed property is mandated by EPUB itself.

## Available validator

No EPUBCheck executable or JAR was found in PATH, the usual Homebrew/local/MacPorts share directories, the standard Maven EPUBCheck location, or the inspected desktop application's contents. `/usr/bin/java` exists, but both `java -version` and `/usr/libexec/java_home -V` report that no Java runtime is installed. No validator was downloaded or installed during this bounded check.

Consequently **EPUBCheck is NOT_RUN** unless a later record explicitly supersedes this finding. The [official EPUBCheck instructions](https://www.w3.org/publishing/epubcheck/docs/running/) describe running the validator against a file and interpreting its warnings/errors. Independent XML/ZIP/link/content checks should be reported under their own names; they are not an EPUBCheck certification. Amazon conversion and physical Kindle rendering also remain unrun until separately exercised.
