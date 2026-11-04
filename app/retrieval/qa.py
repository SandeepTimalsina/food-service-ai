import os

from dotenv import load_dotenv
from google import genai

from app.ingestion.vector_store import search_documents

load_dotenv()

# Initialize Gemini
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = os.getenv("GEMINI_MODEL")


def answer_question(question: str) -> str:
    """
    Answer a question using information retrieved from ChromaDB.
    """

    # Retrieve relevant document chunks
    results = search_documents(
        question,
        number_of_results=3
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "No relevant information was found."

    # Build context with source information
    context_parts = []

    for index, document in enumerate(documents):
        source = metadatas[index].get("source", "Unknown")
        chunk = metadatas[index].get("chunk_number", "Unknown")

        context_parts.append(
            f"Source: {source}, Chunk: {chunk}\n{document}"
        )

    context = "\n\n".join(context_parts)

    # Construct RAG prompt
    prompt = f"""
You are a helpful restaurant information assistant.

Answer the user's question using ONLY the information
provided in the retrieved document context.

If the answer is not available in the context,
clearly say that the documents do not provide that information.

Do not invent menu items, ingredients, or prices.

Retrieved document context:
{context}

User question:
{question}

Provide a clear and concise answer.
Mention the relevant source document and chunk number.
"""

    # Generate answer with Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


if __name__ == "__main__":

    question = "Which dishes contain chicken?"

    print(f"\nQuestion: {question}\n")

    answer = answer_question(question)

    print("AI Assistant Answer:")
    print(answer)
