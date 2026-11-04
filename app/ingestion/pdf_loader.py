import pymupdf


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF document.
    """
    document = pymupdf.open(file_path)

    extracted_text = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        if text.strip():
            extracted_text.append(
                f"\n--- Page {page_number} ---\n{text}"
            )

    document.close()

    return "\n".join(extracted_text)


if __name__ == "__main__":
    pdf_path = "data/uploads/chimney-food-menu-2025.pdf"

    text = extract_text_from_pdf(pdf_path)

    print(text)
