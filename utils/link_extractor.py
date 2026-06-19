from PyPDF2 import PdfReader


def extract_pdf_links(pdf_path: str) -> list[str]:
    """
    Extract all hyperlink URIs from a PDF via /Annots -> /A -> /URI.
    Returns deduped list preserving order.
    """
    reader = PdfReader(pdf_path)
    urls: list[str] = []

    for page in reader.pages:
        annots = page.get("/Annots") or []
        for a in annots:
            try:
                obj = a.get_object()
                action = obj.get("/A") if obj else None
                uri = action.get("/URI") if action else None
                if uri:
                    urls.append(str(uri).strip())
            except Exception:
                continue

    # dedupe preserve order
    seen = set()
    out: list[str] = []
    for u in urls:
        u = u.rstrip(".,;")
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out