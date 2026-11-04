from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Split extracted document text into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(text)

    return chunks


if __name__ == "__main__":
    from pdf_loader import extract_text_from_pdf

    pdf_path = "data/uploads/chimney-food-menu-2025.pdf"

    extracted_text = extract_text_from_pdf(pdf_path)

    chunks = split_text_into_chunks(extracted_text)

    print(f"Total chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)
