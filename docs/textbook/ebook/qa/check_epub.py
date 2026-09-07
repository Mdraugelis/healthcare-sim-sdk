#!/usr/bin/env python3
"""Bounded EPUB structure, local-link and source-content audit (not EPUBCheck)."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import posixpath
import re
import struct
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
import zipfile

from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
}
EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"


def normalized(text):
    return " ".join(text.split())


def lexical_text(text):
    """Ignore markup-only whitespace around punctuation, preserving token order."""
    return " ".join(re.findall(r"\w+|[^\w\s]", text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epub", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name(
        "structure_results.json"))
    args = parser.parse_args()
    checks = []

    def check(name, success, detail):
        checks.append({"id": name, "status": "PASS" if success else "FAIL",
                       "detail": detail})

    with zipfile.ZipFile(args.epub) as book:
        names = book.namelist()
        name_set = set(names)
        check("zip_crc", book.testzip() is None, "Archive CRC inspection")
        check("unique_archive_entries", len(names) == len(name_set), len(names))
        check("relative_archive_paths", all(
            not n.startswith("/") and ".." not in n.split("/") for n in names
        ), "No absolute or traversing archive entry paths")
        check("mimetype_first", names[0] == "mimetype", names[0])
        mime = book.getinfo("mimetype")
        check("mimetype_uncompressed", mime.compress_type == zipfile.ZIP_STORED,
              mime.compress_type)
        check("mimetype_no_extra", not mime.extra, len(mime.extra))
        with args.epub.open("rb") as stream:
            header = struct.unpack("<IHHHHHIIIHH", stream.read(30))
        check("mimetype_local_header", header[0] == 0x04034B50
              and header[3] == 0 and header[-1] == 0,
              {"compression": header[3], "extra_length": header[-1]})
        check("mimetype_exact", book.read("mimetype") == b"application/epub+zip",
              book.read("mimetype").decode("ascii", errors="replace"))
        check("send_to_kindle_web_size", args.epub.stat().st_size <= 200_000_000,
              args.epub.stat().st_size)
        container = ET.fromstring(book.read("META-INF/container.xml"))
        rootfiles = container.findall("container:rootfiles/container:rootfile", NS)
        check("container_rootfile", len(rootfiles) == 1,
              [r.attrib for r in rootfiles])
        package_path = rootfiles[0].attrib["full-path"]
        package = ET.fromstring(book.read(package_path))
        check("epub3_package", package.attrib.get("version") == "3.0",
              package.attrib)
        manifest = package.findall("opf:manifest/opf:item", NS)
        items = {item.attrib["id"]: item for item in manifest}
        check("manifest_ids_unique", len(items) == len(manifest), len(manifest))

        def resolve(source, reference):
            uri = urlsplit(reference)
            if uri.scheme or uri.netloc:
                return None, unquote(uri.fragment)
            target = (posixpath.normpath(posixpath.join(
                posixpath.dirname(source), unquote(uri.path),
            )) if uri.path else source)
            return target, unquote(uri.fragment)

        paths = {key: resolve(package_path, item.attrib["href"])[0]
                 for key, item in items.items()}
        check("manifest_files_exist", all(p in name_set for p in paths.values()),
              [p for p in paths.values() if p not in name_set])
        check("manifest_hrefs_unique", len(set(paths.values())) == len(paths),
              len(paths))
        metadata = package.find("opf:metadata", NS)
        uid = package.attrib.get("unique-identifier")
        identifiers = metadata.findall("dc:identifier", NS)
        check("unique_identifier_resolves", any(
            d.attrib.get("id") == uid and normalized(d.text or "")
            for d in identifiers
        ), uid)
        for field in ["title", "language"]:
            element = metadata.find(f"dc:{field}", NS)
            check(f"metadata_{field}", element is not None and bool(element.text),
                  element.text if element is not None else None)
        modified = [m.text for m in metadata.findall("opf:meta", NS)
                    if m.attrib.get("property") == "dcterms:modified"]
        check("modified_timestamp", len(modified) == 1 and bool(re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", modified[0] or "",
        )), modified)
        spine = package.findall("opf:spine/opf:itemref", NS)
        spine_ids = [item.attrib["idref"] for item in spine]
        check("spine_manifest_refs", bool(spine_ids) and all(
            key in items for key in spine_ids
        ), spine_ids)
        check("spine_unique", len(spine_ids) == len(set(spine_ids)), spine_ids)
        spine_paths = [paths[key] for key in spine_ids]
        nav_ids = [key for key, item in items.items()
                   if "nav" in item.attrib.get("properties", "").split()]
        check("one_navigation_document", len(nav_ids) == 1, nav_ids)

        documents = {}
        id_sets = {}
        for key, item in items.items():
            if item.attrib["media-type"] in ["application/xhtml+xml", "image/svg+xml"]:
                path = paths[key]
                tree = ET.fromstring(book.read(path))
                documents[path] = tree
                ids = [n.attrib["id"] for n in tree.iter() if "id" in n.attrib]
                id_sets[path] = set(ids)
                check("ids_unique:" + path, len(ids) == len(set(ids)), len(ids))
                if item.attrib["media-type"] == "application/xhtml+xml":
                    check("xhtml_namespace:" + path,
                          tree.tag == "{http://www.w3.org/1999/xhtml}html", tree.tag)
        nav_path = paths[nav_ids[0]]
        toc = [n for n in documents[nav_path].iter()
               if n.tag == "{http://www.w3.org/1999/xhtml}nav"
               and "toc" in n.attrib.get(EPUB_TYPE, "").split()]
        check("one_toc_nav", len(toc) == 1, len(toc))
        nav_targets = [resolve(nav_path, n.attrib["href"])[0]
                       for n in toc[0].iter() if "href" in n.attrib]
        check("toc_content_in_spine", all(
            target is None or target in spine_paths for target in nav_targets
        ), nav_targets)

        broken = []
        local_link_count = 0
        external_link_count = 0
        forbidden = []
        missing_alt = []
        for path, tree in documents.items():
            for node in tree.iter():
                tag = node.tag.rsplit("}", 1)[-1]
                if tag in {"script", "form", "input", "button", "select",
                           "textarea", "details"}:
                    forbidden.append([path, tag])
                if tag == "img" and not node.attrib.get("alt", "").strip():
                    missing_alt.append([path, node.attrib.get("src")])
                for attr, value in node.attrib.items():
                    if attr.rsplit("}", 1)[-1] not in {"href", "src", "poster"}:
                        continue
                    target, fragment = resolve(path, value)
                    if target is None:
                        external_link_count += 1
                        if urlsplit(value).scheme in {"file", "javascript"}:
                            broken.append([path, value, "nonportable or active scheme"])
                        continue
                    local_link_count += 1
                    if target not in name_set:
                        broken.append([path, value, "missing resource"])
                    elif fragment and fragment not in id_sets.get(target, set()):
                        broken.append([path, value, "missing fragment"])
        for key, item in items.items():
            if item.attrib["media-type"] == "text/css":
                path = paths[key]
                css = book.read(path).decode("utf-8")
                for value in re.findall(r"url\(['\"]?([^'\")]+)", css):
                    target, _ = resolve(path, value)
                    if target is not None and target not in name_set:
                        broken.append([path, value, "missing CSS resource"])
        check("local_links_resolve", not broken, broken)
        check("static_reading_no_required_widgets", not forbidden, forbidden)
        check("image_alternatives", not missing_alt, missing_alt)

        images = []
        for key, item in items.items():
            media = item.attrib["media-type"]
            if media.startswith("image/") and media != "image/svg+xml":
                with Image.open(BytesIO(book.read(paths[key]))) as im:
                    dimensions = list(im.size)
                    im.verify()
                images.append({"path": paths[key], "dimensions": dimensions})
        check("raster_images_decode", bool(images), images)

        # Compare source units, not just word totals: a missing paragraph can be
        # hidden by newly added static lab text in a total-count comparison.
        source = BeautifulSoup((ROOT / "docs/textbook/index.html").read_text(),
                               "html.parser")
        numbered = [c for c in source.select(".chapter")
                    if re.match(r"^\d\d-", c["id"])]
        toc_links = [resolve(nav_path, n.attrib["href"])
                     for n in toc[0].iter() if "href" in n.attrib]
        chapter_targets = ["book-" + c["id"] for c in numbered]
        toc_chapters = [fragment for _, fragment in toc_links
                        if fragment in chapter_targets]
        check("sixteen_chapters_in_toc_order", toc_chapters == chapter_targets,
              {"expected": chapter_targets, "actual": toc_chapters})

        source_code = Counter(p.get_text().strip()
                              for p in source.select("details.code-reference pre"))
        ebook_code = Counter("".join(n.itertext()).strip()
                             for tree in documents.values() for n in tree.iter()
                             if n.tag.endswith("}pre"))
        missing_code = source_code - ebook_code
        check("all_cited_code_excerpts_exact", not missing_code,
              {"expected_instances": sum(source_code.values()),
               "missing_instances": sum(missing_code.values()),
               "missing_prefixes": [x[:180] for x in missing_code]})

        # Remove only converter-marked, added printed path annotations in the
        # in-memory comparison copy. They remain in the actual EPUB.
        printed_annotations = []
        for tree in documents.values():
            for n in tree.iter():
                if "repository-reference" in n.attrib.get("class", "").split():
                    original = n.text or ""
                    clean = re.sub(r" \[[^\]]+\]$", "", original)
                    if clean != original:
                        printed_annotations.append(original)
                        n.text = clean
        text = lexical_text(" ".join(
            "".join(documents[p].itertext()) for p in spine_paths
            if p in documents
        ))
        content = []
        omitted_web_fallbacks = 0
        for chapter in source.select(".chapter"):
            units = []
            for node in chapter.select("p, li, td, th, h2, h3, h4, pre, figcaption"):
                if node.find_parent("svg"):
                    continue
                fallback = normalized(node.get_text(" ", strip=True))
                if (node.find_parent(attrs={"data-lab": True})
                        and fallback == (
                            "Interactive teaching calculator. Enable JavaScript "
                            "to vary inputs; the surrounding chapter gives the "
                            "formula, assumptions and worked example."
                        )):
                    omitted_web_fallbacks += 1
                    continue
                unit = lexical_text(node.get_text(" ", strip=True))
                if unit:
                    units.append(unit)
            missing = [unit for unit in units if unit not in text]
            content.append({"chapter": chapter["id"], "source_units": len(units),
                            "matched_units": len(units) - len(missing),
                            "unmatched_units": missing})
        check("sixteen_source_chapters", len(numbered) == 16, len(numbered))
        # Unmatched units need review: intentional web-to-print adaptations may
        # explain them, but this checker never silently assumes that they do.
        check("source_content_units_preserved", all(
            not item["unmatched_units"] for item in content
        ), [{"chapter": c["chapter"], "unmatched_units": c["unmatched_units"]}
            for c in content if c["unmatched_units"]])
        for path in sorted((ROOT / "docs/textbook/chapters").glob("*.json")):
            chapter = json.loads(path.read_text())
            required_text = [chapter["title"], *chapter["objectives"],
                             *chapter["review_questions"]]
            for exercise in chapter["exercises"]:
                required_text.extend([exercise["question"], exercise["answer"]])
            missing = [s for s in required_text if lexical_text(s) not in text]
            check("learning_material:" + chapter["id"], not missing, missing)

        assets = json.loads((ROOT / "docs/textbook/ebook/assets_manifest.json")
                            .read_text())
        missing_notation = [e["id"] for e in assets["equations"]
                            if e["latex"].strip() not in ebook_code]
        check("literal_notation_appendix_preserved", not missing_notation,
              {"expected": len(assets["equations"]),
               "missing": missing_notation})
        image_hashes = {hashlib.sha256(book.read(paths[key])).hexdigest()
                        for key, item in items.items()
                        if item.attrib["media-type"].startswith("image/")}
        expected_images = [e["image"] for group in ["equations", "diagrams"]
                           for e in assets[group]
                           if not e.get("key", "").startswith("hero-")]
        expected_images.extend(chart["image"] for lab in assets["labs"]
                               for chart in lab["charts"])
        hero_versions = [e["image"] for e in assets["diagrams"]
                         if e.get("key", "").startswith("hero-")]
        check("hero_diagram_version_packaged", any(
            e["sha256"] in image_hashes for e in hero_versions
        ), [e["path"] for e in hero_versions if e["sha256"] in image_hashes])
        missing_images = [e["path"] for e in expected_images
                          if e["sha256"] not in image_hashes]
        check("exported_visuals_packaged_exact", not missing_images,
              {"expected": len(expected_images), "missing": missing_images})
        check("static_labs_replace_web_fallbacks",
              len(assets["labs"]) == omitted_web_fallbacks == 8,
              {"static_labs": len(assets["labs"]),
               "omitted_web_fallbacks": omitted_web_fallbacks})

    report = {
        "status": "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS",
        "epub": str(args.epub.resolve()),
        "epub_sha256": hashlib.sha256(args.epub.read_bytes()).hexdigest(),
        "source_html_sha256": hashlib.sha256(
            (ROOT / "docs/textbook/index.html").read_bytes(),
        ).hexdigest(),
        "check_count": len(checks), "checks": checks,
        "link_counts": {"local": local_link_count, "external": external_link_count},
        "chapter_content": content,
        "content_normalizations": {
            "printed_path_annotations": len(printed_annotations),
            "web_only_fallback_notices": omitted_web_fallbacks,
            "rule": "Ignore markup whitespace and declared added path annotations; "
                    "preserve source unit tokens and exact cited code excerpts.",
        },
        "spine": spine_paths,
        "epubcheck": "NOT_RUN: no available EPUBCheck/Java installation",
        "kindle_conversion": "NOT_RUN: no account upload or device delivery",
        "limitations": ["Bounded checks, not complete EPUB conformance validation",
                        "Image readability and layout need separate visual inspection"],
    }
    args.output.write_text(json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "failures": Counter(c["status"] for c in checks)["FAIL"],
                      "output": str(args.output)}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
