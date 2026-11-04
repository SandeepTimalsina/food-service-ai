import chromadb
from sentence_transformers import SentenceTransformer

from app.ingestion.pdf_loader import extract_text_from_pdf
from app.ingestion.text_splitter import split_text_into_chunks


# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a persistent ChromaDB database
client = chromadb.PersistentClient(path="data/chroma_db")

collection = client.get_or_create_collection(
    name="food_service_documents"
)


def store_pdf(pdf_path: str):
    """Extract, split, embed, and store PDF content."""

    # Extract PDF text
    text = extract_text_from_pdf(pdf_path)

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    if not chunks:
        print("No text found in the PDF.")
        return

    # Generate embeddings
    embeddings = embedding_model.encode(chunks).tolist()

    # Prepare unique IDs and metadata
    ids = [
        f"{pdf_path.replace('/', '_')}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": pdf_path,
            "chunk_number": i + 1
        }
        for i in range(len(chunks))
    ]

    # Store in ChromaDB
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully stored {len(chunks)} chunks in ChromaDB.")


def search_documents(question: str, number_of_results: int = 3):
    """Find document chunks relevant to a question."""

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=number_of_results
    )

    return results


if __name__ == "__main__":

    pdf_path = "data/uploads/chimney-food-menu-2025.pdf"

    # Store the PDF
    store_pdf(pdf_path)

    # Test semantic search
    question = "Which dishes contain chicken?"

    print(f"\nQuestion: {question}")

    results = search_documents(question)

    for i, document in enumerate(results["documents"][0], start=1):
        print(f"\n--- Result {i} ---")
        print(document)
