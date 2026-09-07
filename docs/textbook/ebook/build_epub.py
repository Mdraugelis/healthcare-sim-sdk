#!/usr/bin/env python3
"""Create a reflowable, static Kindle/EPUB edition of the reviewed HTML book.

Prerequisites: export_assets.cjs, Python bs4/Pillow, and Pandoc.
Does not modify the original HTML edition or send anything to Amazon.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent
ROOT = BOOK.parents[1]
TITLE = "The Science of Simulating Healthcare AI"
EDITION = "2026-09-06"
OUT = HERE / "healthcare-simulation-sdk-textbook.epub"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cover():
    """Draw an original typographic cover; no external image assets."""
    image = Image.new("RGB", (1600, 2560), "#f5f3ed")
    draw = ImageDraw.Draw(image)
    serif = "/System/Library/Fonts/Supplemental/Georgia.ttf"
    sans = "/System/Library/Fonts/Supplemental/Arial.ttf"
    if not Path(serif).exists():
        serif = "DejaVuSerif.ttf"
        sans = "DejaVuSans.ttf"
    titlefont = ImageFont.truetype(serif, 142)
    medium = ImageFont.truetype(sans, 46)
    small = ImageFont.truetype(sans, 30)
    label = ImageFont.truetype(sans, 37)
    ink, teal, line = "#172a30", "#086b65", "#c5d5ce"
    draw.rectangle((0, 0, 1600, 28), fill=teal)
    draw.text((138, 160), "HEALTHCARE SIMULATION SDK", font=label, fill=teal)
    y = 395
    for text in ["The Science", "of Simulating", "Healthcare AI"]:
        draw.text((132, y), text, font=titlefont, fill=ink)
        y += 187
    draw.line((138, 1050, 1462, 1050), fill=line, width=3)
    for i, text in enumerate(["Probability, models, decisions", "and the evidence behind them"]):
        draw.text((138, 1120 + i * 66), text, font=medium, fill=ink)
    # A branching motif mirrors the book's factual/comparator design.
    draw.rounded_rectangle((140, 1555, 550, 1745), radius=12, fill="#dcece6")
    draw.text((182, 1620), "Population", font=medium, fill=ink)
    draw.line((550, 1650, 720, 1650), fill=teal, width=7)
    draw.line((720, 1470, 720, 1830), fill=teal, width=7)
    draw.line((720, 1470, 880, 1470), fill=teal, width=7)
    draw.line((720, 1830, 880, 1830), fill=teal, width=7)
    draw.rounded_rectangle((880, 1380, 1460, 1560), radius=12, fill="#dcece6")
    draw.rounded_rectangle((880, 1740, 1460, 1920), radius=12, fill="#e9dfcc")
    draw.text((925, 1438), "Factual policy", font=medium, fill=ink)
    draw.text((925, 1798), "Comparator", font=medium, fill=ink)
    draw.text((138, 2170), "AN ILLUSTRATED TECHNICAL TEXTBOOK", font=label, fill=teal)
    draw.text((138, 2255), "16 chapters · worked examples · source notes", font=small, fill=ink)
    draw.text((138, 2395), "Kindle reading edition · September 2026", font=small, fill=ink)
    image.save(HERE / "cover.png", optimize=True)


def main():
    assets = json.loads((HERE / "assets_manifest.json").read_text())
    manuscript = json.loads((BOOK / "source_manifest.json").read_text())
    if assets["source_html"]["sha256"] != sha(BOOK / "index.html"):
        raise ValueError("Visual export is stale: export current HTML before building")
    source = BeautifulSoup((BOOK / "index.html").read_text(), "html.parser")
    output = BeautifulSoup('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/><title></title></head><body></body></html>', 'html.parser')
    output.title.string = TITLE
    css = output.new_tag('link', rel='stylesheet', href='ebook.css')
    output.head.append(css)
    body = output.body

    def fragment(markup):
        return BeautifulSoup(markup, 'html.parser')

    def append_markup(markup, parent=body):
        for child in list(fragment(markup).contents):
            parent.append(child)

    def paragraph(text, parent, klass=None):
        p = output.new_tag('p')
        if klass:
            p['class'] = klass
        p.string = text
        parent.append(p)
        return p

    def picture(item, alt, klass):
        img = output.new_tag('img', src=item['path'], alt=alt)
        img['class'] = klass
        # Width at half the CSS capture width is not appropriate: use recorded CSS px.
        # CSS clamps long formulas to the reading column; high-res raster stays sharp.
        if klass == 'equation-image':
            img['style'] = f"width:{round(item['css_width'])}px;max-width:100%;height:auto"
        return img

    append_markup(f'''<section id="reading-edition"><h1>{TITLE}</h1>
<p class="subtitle">Probability, models, decisions and the evidence behind them</p>
<p>Healthcare Simulation SDK project documentation</p>
<p>Kindle reading edition · {EDITION}<br/>Original manuscript edition · 2026-09-05</p>
<p><strong>Reviewable author draft.</strong> External scientific peer review and institution-specific clinical validation remain pending.</p>
<h2>About this edition</h2><p>This is the project textbook converted into a reflowable e-book. All sixteen chapters, worked exercises, references and cited source excerpts are retained. Text can follow your reader’s font and size settings. The cover and scientific figures are embedded images.</p>
<p>The eight interactive labs become static worked examples: each records its default inputs, charts, numerical results and assumptions. References to sliders, browser search, print controls or local review notes elsewhere in the text describe the original HTML companion. Those controls do not run in this e-book.</p>
<p>Tables are presented as labeled records so wide comparisons remain readable on a narrow screen. Exercise answers are expanded. The detailed code excerpts follow the main chapters as source-note appendices; mathematical source notation is retained in its own appendix. Repository file links are printed references because local workspace files are not available on a Kindle. External scholarly links remain clickable when your reader has network access.</p>
<h2>Reading paths</h2><ul><li>Clinical and operations: chapters 01, 08, 12 and 16.</li><li>Quantitative and research: chapters 02–05, then 09–11.</li><li>Engineering and assurance: chapters 07 and 13–16, followed by the source notes.</li></ul>
</section>''')
    hero = next(d for d in assets['diagrams'] if d['key'] == 'hero-compact') if any(d['key']=='hero-compact' for d in assets['diagrams']) else assets['diagrams'][0]
    body.section.append(picture(hero['image'], hero['description'], 'diagram-image'))
    paragraph('A shared modeled population supports two policy worlds. The paired contrast remains conditional on the model assumptions.', body.section, 'caption')
    equations = {e['id']: e for e in assets['equations']}
    labs = {e['key']: e for e in assets['labs']}
    source_appendices = []
    notation = []
    chapters = list(source.select('.chapter'))
    counts = {'chapters': 0, 'equations': 0, 'labs': 0, 'code_excerpts': 0, 'tables': 0}
    for chapter in chapters:
        ch = copy.deepcopy(chapter)
        chapter_id = ch.get('id', '')
        if re.match(r'^\d\d-', chapter_id):
            counts['chapters'] += 1
        # Preserve every code excerpt, but move it behind the main reading sequence.
        code_refs = list(ch.select('details.code-reference'))
        if code_refs:
            appendix = output.new_tag('section', id='sources-'+chapter_id)
            h = output.new_tag('h1')
            h.string = f"Source notes {chapter_id[:2]}: {ch.h2.get_text(' ', strip=True)}"
            appendix.append(h)
            paragraph('These excerpts are from the reviewed HTML edition. File hashes identify the full source files; excerpts marked as opening selections are not complete definitions.', appendix)
            for ref in code_refs:
                appendix.append(ref.extract())
                counts['code_excerpts'] += 1
            source_appendices.append(appendix)
            link = output.new_tag('a', href='#sources-'+chapter_id)
            link.string = f"Read the source notes for Chapter {chapter_id[:2]}"
            p = output.new_tag('p', attrs={'class':'source-link'})
            p.append(link)
            target = ch.find('h3', string='Trace the implementation')
            if target:
                target.insert_after(p)
            else:
                ch.append(p)
        for figure in ch.select('figure.equation'):
            record = equations[figure['id']]
            figure.svg.replace_with(picture(record['image'], record['latex'], 'equation-image'))
            details = figure.select_one('.equation-source')
            if details:
                details.decompose()
            notation.append(record)
            link = output.new_tag('a', href='#notation-'+record['id'])
            link.string = 'Mathematical source notation'
            p = output.new_tag('p', attrs={'class':'caption'})
            p.append(link)
            figure.append(p)
            counts['equations'] += 1
        for holder in ch.select('[data-lab]'):
            lab = labs[holder['data-lab']]
            box = output.new_tag('div', attrs={'class':'static-lab'})
            h = output.new_tag('h3')
            h.string = 'Worked lab: ' + lab['title']
            box.append(h)
            paragraph(lab['description'], box)
            paragraph('Static example from the HTML lab’s default settings.', box, 'caption')
            inputs = output.new_tag('ul')
            for item in lab['default_parameters']:
                li = output.new_tag('li')
                li.string = f"{item['label']}: {item['display_value']}"
                inputs.append(li)
            box.append(inputs)
            for chart in lab['charts']:
                fig = output.new_tag('figure')
                fig.append(picture(chart['image'], chart['alt'], 'chart-image'))
                cap = output.new_tag('figcaption')
                cap.string = chart['title']
                fig.append(cap)
                box.append(fig)
            paragraph(lab['results_text'], box, 'lab-result')
            for note in lab['notes']:
                paragraph(note, box, 'caption')
            for table in lab.get('tables', []):
                t = output.new_tag('table')
                head = output.new_tag('tr')
                for value in table['headers']:
                    th = output.new_tag('th'); th.string = value; head.append(th)
                t.append(head)
                for row in table['rows']:
                    tr = output.new_tag('tr')
                    for value in row:
                        td = output.new_tag('td'); td.string = str(value); tr.append(td)
                    t.append(tr)
                box.append(t)
            holder.replace_with(box)
            counts['labs'] += 1
        # The additional engine diagram is authored as SVG in chapter 14.
        for svg in list(ch.select('svg')):
            record = next(d for d in assets['diagrams'] if d['key'] == 'engine-branches')
            svg.replace_with(picture(record['image'], record['description'], 'diagram-image'))
        # Flatten heading hierarchy so the EPUB has one primary chapter per file.
        for heading in ch.find_all(re.compile('^h[2-6]$')):
            heading.name = 'h' + str(int(heading.name[1]) - 1)
        if chapter_id in ('glossary','bibliography','evidence'):
            # These sections have no chapter header, but have an h2 in original HTML.
            pass
        body.append(ch)
    for appendix in source_appendices:
        body.append(appendix)
    notation_section = output.new_tag('section', id='equation-notation')
    h = output.new_tag('h1'); h.string = 'Mathematical notation appendix'; notation_section.append(h)
    paragraph('The following LaTeX strings preserve the exact source of each rendered equation for inspection and transcription. They are notation, not executable code.', notation_section)
    for eq in notation:
        h = output.new_tag('h2', id='notation-'+eq['id']); h.string = eq['label']; notation_section.append(h)
        pre = output.new_tag('pre'); pre.string = eq['latex']; notation_section.append(pre)
    body.append(notation_section)
    # Details/summary relies on UI behavior; expand every answer, reference and note.
    for details in list(body.select('details')):
        details.name = 'div'
        details['class'] = list(details.get('class', [])) + ['expanded-note']
        summary = details.find('summary', recursive=False)
        if summary:
            summary.name = 'h3'
    # No wide table depends on horizontal scrolling on an e-ink screen.
    for table in list(body.select('table')):
        rows = table.find_all('tr')
        if not rows:
            table.decompose(); continue
        headers = [c.get_text(' ', strip=True) for c in rows[0].find_all(['th','td'], recursive=False)]
        cards = output.new_tag('div', attrs={'class':'comparison-records'})
        for row in rows[1:]:
            cells = row.find_all(['th','td'], recursive=False)
            card = output.new_tag('div', attrs={'class':'comparison-record'})
            for index, cell in enumerate(cells):
                p = output.new_tag('p')
                label = output.new_tag('strong')
                label.string = (headers[index] if index < len(headers) else f'Field {index+1}') + ': '
                p.append(label)
                for child in list(cell.contents):
                    p.append(child.extract())
                card.append(p)
            cards.append(card)
        table.replace_with(cards)
        counts['tables'] += 1
    # Repository-relative resources cannot be resolved on a standalone Kindle.
    local_refs = []
    for a in list(body.select('a[href]')):
        href = a['href']
        if href.startswith(('#','https://','http://','mailto:')):
            continue
        local_refs.append(href)
        span = output.new_tag('span', attrs={'class':'repository-reference'})
        for child in list(a.contents):
            span.append(child.extract())
        ref = href.removeprefix('../../')
        if ref not in span.get_text():
            span.append(' ['+ref+']')
        a.replace_with(span)
    for el in body.select('[style]'):
        if el.name != 'img':
            del el['style']
    for el in body.select('[hidden]'):
        del el['hidden']
    for el in body.select('script,button,input,select,textarea,noscript'):
        el.decompose()
    # Rewrite numeric/HTML identifiers into robust XML-friendly identifiers.
    id_map = {el['id']:'book-'+el['id'] for el in body.select('[id]')}
    for el in body.select('[id]'):
        el['id'] = id_map[el['id']]
    for a in body.select('a[href^="#"]'):
        ident = a['href'][1:]
        if ident not in id_map:
            raise ValueError('Unresolved e-book anchor: '+ident)
        a['href'] = '#'+id_map[ident]
    for el in body.select('[aria-labelledby]'):
        el['aria-labelledby'] = ' '.join(id_map.get(v,v) for v in el['aria-labelledby'].split())
    appendix = output.new_tag('section', id='ebook-edition-record')
    h = output.new_tag('h1'); h.string = 'E-book edition record'; appendix.append(h)
    paragraph(f'This reading edition was converted on {EDITION} from the reviewed HTML manuscript. It changes presentation, not the simulation implementation or scientific claims.', appendix)
    paragraph('Manuscript identity: '+manuscript['manuscript_identity'], appendix, 'identity')
    paragraph('Original HTML SHA-256: '+sha(BOOK/'index.html'), appendix, 'identity')
    paragraph('The original HTML was checked with 137 numerical comparisons and 30 Chromium checks. Those are the HTML documentation checks, not evidence of execution on a Kindle. EPUB packaging and static-reading checks are recorded separately in the companion e-book QA report.', appendix)
    paragraph('The EPUB contains no scripts, live review-note storage, personal browser notes or institutional patient data. The supplied examples are synthetic project demonstrations. This file has not been uploaded to an Amazon account by its builder.', appendix)
    body.append(appendix)
    # Pandoc only splits/registers headings that are not buried in HTML containers.
    # Promote chapter identifiers onto their headings before unwrapping sections.
    for section in list(body.find_all('section')):
        heading = section.find('h1')
        if heading and section.get('id'):
            old = heading.get('id')
            heading['id'] = section['id']
            if old and old != section['id']:
                anchor = output.new_tag('span', id=old)
                heading.insert_before(anchor)
        section.unwrap()
    for container in list(body.select('header, .chapter-body')):
        container.unwrap()
    cover()
    reading = HERE / 'reading_copy.html'
    reading.write_text(str(output), encoding='utf-8')
    pandoc = shutil.which('pandoc')
    if not pandoc:
        raise RuntimeError('Pandoc is required to build the EPUB')
    identity = str(uuid.uuid5(uuid.NAMESPACE_URL, manuscript['manuscript_identity']+'/kindle/'+EDITION))
    command = [pandoc, str(reading), '--from=html', '--to=epub3', '--standalone',
               '--toc', '--toc-depth=1', '--split-level=1', '--epub-title-page=false',
               '--syntax-highlighting=none',
               '--css='+str(HERE/'ebook.css'), '--epub-cover-image='+str(HERE/'cover.png'),
               '--resource-path='+str(HERE), '--metadata=title:'+TITLE,
               '--metadata=author:Healthcare Simulation SDK project', '--metadata=lang:en-US',
               '--metadata=date:'+EDITION, '--metadata=identifier:urn:uuid:'+identity,
               '--output='+str(OUT)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    if result.stderr:
        print(result.stderr)
    manifest = {'edition_date':EDITION, 'title':TITLE, 'source_html_sha256':sha(BOOK/'index.html'),
                'manuscript_identity':manuscript['manuscript_identity'], 'epub_sha256':sha(OUT),
                'epub_bytes':OUT.stat().st_size, 'content_counts':counts,
                'repository_links_converted_to_printed_references':len(local_refs),
                'builder_sha256':sha(Path(__file__)), 'assets_manifest_sha256':sha(HERE/'assets_manifest.json'),
                'pandoc':subprocess.check_output([pandoc,'--version'],text=True).splitlines()[0],
                'pandoc_warnings':result.stderr, 'uploaded_to_amazon':False}
    (HERE/'edition_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
