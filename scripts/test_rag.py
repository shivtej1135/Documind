import sys
from pathlib import Path

# Add the project root so we can import modules from src/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.retrieval.retriever import Retriever
from src.generation.answer_generator import AnswerGenerator


def main():
    # Create the retriever and load the FAISS index.
    retriever = Retriever()

    # Create the LLM answer generator.
    generator = AnswerGenerator()

    # Test question for our citation-enabled RAG pipeline.
    question = (
        "How many layers did the deepest residual network "
        "presented in the paper have?"
    )

    print(f"Question: {question}")
    print("\nRetrieving relevant chunks...")

    # Retrieve the top 5 chunks related to the question.
    retrieved_chunks = retriever.retrieve(
        question,
        top_k=5,
    )

    print(f"Retrieved {len(retrieved_chunks)} chunks")

    print("\nGenerating answer...")

    # Generate the answer and attach metadata-derived citations.
    result = generator.generate(
        question,
        retrieved_chunks,
    )

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    # Print every source used to build the context.
    for source in result["sources"]:
        print(
            f"- Document: {source['doc_id']}, "
            f"Page: {source['page_num']}, "
            f"Type: {source['chunk_type']}, "
            f"Section: {source['section_heading']}"
        )

    print("\nHallucination Check:")

    if result["hallucination_check"]["passed"]:
        print("Passed - all numbers in the answer were found in the context.")
    else:
        print("Warning - unsupported numbers were detected:")

        for number in result["hallucination_check"]["unsupported_numbers"]:
            print(f"- {number}")


if __name__ == "__main__":
    main()