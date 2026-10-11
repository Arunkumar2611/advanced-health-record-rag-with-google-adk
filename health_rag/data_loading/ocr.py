from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium


MAX_SCANNED_PDF_PAGES = 25


def render_scanned_pdf(path: Path) -> tuple[list[bytes], int]:
    """Render image-only PDF pages for Gemini vision OCR, with a clear page cap."""
    pdf = pdfium.PdfDocument(str(path))
    page_count = len(pdf)
    if page_count > MAX_SCANNED_PDF_PAGES:
        raise ValueError(f"Scanned PDF has {page_count} pages; maximum supported is {MAX_SCANNED_PDF_PAGES}")
    rendered = []
    for index in range(page_count):
        page = pdf[index]
        bitmap = page.render(scale=1.8)
        image = bitmap.to_pil().convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        rendered.append(buffer.getvalue())
        page.close()
    pdf.close()
    return rendered, page_count
