"""Render the Markdown RAG proposal into a reviewable PDF.

Dependencies: pymupdf, markdown.
Run from any working directory: python project/scripts/build_proposal_pdf.py
"""

from pathlib import Path
import re

import fitz
import markdown


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "project" / "proposal" / "5001_Proposal_RAG.md"
OUTPUT = ROOT / "output" / "pdf" / "5001_Proposal_RAG.pdf"
TEMP = ROOT / "tmp" / "pdfs" / "rag_proposal"

CSS = """
body { font-family: sans-serif; font-size: 10.1pt; line-height: 1.20;
       color: #202b38; }
h1 { font-size: 20pt; color: #123d62; margin-top: 0; margin-bottom: 9pt; }
h2 { font-size: 11.6pt; color: #123d62; margin-top: 10pt; margin-bottom: 4pt;
     page-break-after: avoid; }
h3 { font-size: 10.8pt; color: #123d62; margin-top: 9pt; margin-bottom: 4pt; }
p { margin-top: 0; margin-bottom: 5pt; }
ul, ol { margin-top: 3pt; margin-bottom: 6pt; padding-left: 18pt; }
li { margin-bottom: 4pt; }
strong { font-weight: bold; }
a { color: #185b85; text-decoration: none; }
table { border-collapse: collapse; width: 100%; margin-top: 5pt;
        margin-bottom: 9pt; font-size: 9.2pt; }
th { color: #123d62; font-weight: bold; }
td, th { border: 0.5pt solid #c6d1da; padding: 3pt 5pt; text-align: left; }
"""


def main():
    TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = SOURCE.read_text(encoding="utf-8")
    text = re.sub("[\u2010-\u2015\u2212]", "-", text)
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    story = fitz.Story(html="<html><body>" + body + "</body></html>", user_css=CSS)
    raw_pdf = TEMP / "proposal_flow.pdf"
    writer = fitz.DocumentWriter(str(raw_pdf))
    media = fitz.paper_rect("a4")
    content = fitz.Rect(44, 44, media.width - 44, media.height - 44)
    page_count = 0
    more = True
    while more:
        page_count += 1
        if page_count > 8:
            raise RuntimeError("Unexpected pagination: inspect Markdown and CSS.")
        device = writer.begin_page(media)
        more, _ = story.place(content)
        story.draw(device)
        writer.end_page()
    writer.close()
    del device, writer

    document = fitz.open(raw_pdf)
    for index, page in enumerate(document):
        page.insert_text(
            (44, 28), "SDSC5001 | RAG retrieval and learned reranking",
            fontsize=8, fontname="helv", color=(0.32, 0.40, 0.47),
        )
        page.draw_line(
            (44, media.height - 34), (media.width - 44, media.height - 34),
            color=(0.76, 0.81, 0.85), width=0.5,
        )
        page.insert_text(
            (44, media.height - 21), "Proposal draft | 7 November 2026",
            fontsize=8, fontname="helv", color=(0.32, 0.40, 0.47),
        )
        page.insert_text(
            (media.width - 83, media.height - 21),
            f"{index + 1} / {len(document)}", fontsize=8, fontname="helv",
            color=(0.32, 0.40, 0.47),
        )
    document.set_metadata({
        "title": "Can Learned Reranking Improve Hybrid Retrieval for RAG?",
        "subject": "SDSC5001 project proposal",
        "creator": "Markdown proposal renderer (PyMuPDF)",
    })
    document.save(OUTPUT, garbage=4, deflate=True)
    document.close()
    try:
        raw_pdf.unlink()
    except PermissionError:
        # Windows can retain a short-lived renderer handle; the file is ignored.
        pass
    final = fitz.open(OUTPUT)
    for index, page in enumerate(final):
        page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(
            TEMP / f"page_{index + 1}.png"
        )
    print(f"PDF: {OUTPUT}")
    print(f"Pages: {len(final)}")
    print(f"Preview: {TEMP}")
    final.close()


if __name__ == "__main__":
    main()
