import json
import re
import sys
from pathlib import Path

# Add the project root so we can import modules from src/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.retrieval.retriever import Retriever
from src.generation.answer_generator import AnswerGenerator


EVAL_FILE = Path("data/eval_set.json")
RESULTS_FILE = Path("data/baseline_results.json")


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "was",
    "were",
    "are",
    "be",
    "that",
    "this",
    "by",
    "as",
    "from",
    "it",
    "its",
    "their",
    "they",
    "than",
    "have",
    "has",
    "had",
}


def normalize_text(text: str) -> str:
    """
    Normalize text so we can perform a simple comparison.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%]+", " ", text)
    return " ".join(text.split())


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful words from an answer while ignoring
    common words that do not help with answer comparison.
    """
    words = normalize_text(text).split()

    return {
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    }


def extract_numbers(text: str) -> set[str]:
    """
    Extract numbers such as 152, 1202, 3.57 and 77.63.
    """
    return set(
        re.findall(
            r"\b\d+(?:\.\d+)?\b",
            text,
        )
    )


def compare_answer(
    expected_answer: str,
    generated_answer: str,
) -> bool:
    """
    Perform a lightweight answer comparison.

    This is a baseline heuristic, not a semantic evaluator.
    """

    expected_normalized = normalize_text(expected_answer)
    generated_normalized = normalize_text(generated_answer)

    # If the complete expected answer appears in the generated
    # answer, we can confidently treat it as a match.
    if expected_normalized in generated_normalized:
        return True

    # Important numbers from the expected answer should appear
    # in the generated answer.
    expected_numbers = extract_numbers(expected_answer)
    generated_numbers = extract_numbers(generated_answer)

    if expected_numbers and not expected_numbers.issubset(
        generated_numbers
    ):
        return False

    # Compare the meaningful words in the expected answer.
    expected_keywords = extract_keywords(expected_answer)
    generated_keywords = extract_keywords(generated_answer)

    if not expected_keywords:
        return False

    matching_keywords = expected_keywords & generated_keywords
    overlap = len(matching_keywords) / len(expected_keywords)

    # Require at least 60% of the important expected words.
    return overlap >= 0.60


def main():
    # Load the fixed evaluation questions.
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} evaluation questions")

    # Load the retriever and LLM once.
    retriever = Retriever()
    generator = AnswerGenerator()

    results = []

    total_page_hits = 0
    total_hallucination_passes = 0
    total_answer_matches = 0

    category_stats = {}

    for index, item in enumerate(eval_set, start=1):
        
        question_id = item["id"]
        category = item["category"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_pages = item["expected_pages"]

        print(
            f"\n[{index}/{len(eval_set)}] "
            f"{question_id}: {question}"
        )

        # Retrieve the top 5 chunks for the question.
        retrieved_chunks = retriever.retrieve(
            question,
            top_k=5,
        )

        # Generate the answer and attach source metadata.
        result = generator.generate(
            question,
            retrieved_chunks,
        )

        generated_answer = result["answer"]

        # Extract the retrieved page numbers.
        retrieved_pages = [
            source["page_num"]
            for source in result["sources"]
            if source["page_num"] is not None
        ]

        # Check whether at least one expected page was retrieved.
        page_hit = any(
            page in expected_pages
            for page in retrieved_pages
        )

        # Check whether the basic numeric grounding test passed.
        hallucination_pass = result["hallucination_check"]["passed"]

        # Compare the generated answer with the expected answer.
        answer_match = compare_answer(
            expected_answer,
            generated_answer,
        )

        if page_hit:
            total_page_hits += 1

        if hallucination_pass:
            total_hallucination_passes += 1

        if answer_match:
            total_answer_matches += 1

        # Create category statistics the first time we see a category.
        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "page_hits": 0,
                "hallucination_passes": 0,
                "answer_matches": 0,
            }

        category_stats[category]["total"] += 1

        if page_hit:
            category_stats[category]["page_hits"] += 1

        if hallucination_pass:
            category_stats[category]["hallucination_passes"] += 1

        if answer_match:
            category_stats[category]["answer_matches"] += 1

        # Save everything needed for later inspection.
        results.append(
            {
                "id": question_id,
                "doc_id": item["doc_id"],
                "category": category,
                "question": question,
                "expected_answer": expected_answer,
                "expected_pages": expected_pages,
                "generated_answer": generated_answer,
                "sources": result["sources"],
                "hallucination_check": result["hallucination_check"],
                "retrieved_pages": retrieved_pages,
                "page_hit": page_hit,
                "answer_match": answer_match,
            }
        )
        print(f"Category: {category}")
        print(f"Page hit: {page_hit}")
        print(f"Answer match: {answer_match}")
        print(f"Hallucination check: {hallucination_pass}")

    # Calculate overall baseline metrics.
    total_questions = len(eval_set)

    page_hit_rate = (
        total_page_hits / total_questions
        if total_questions
        else 0
    )

    hallucination_pass_rate = (
        total_hallucination_passes / total_questions
        if total_questions
        else 0
    )

    answer_match_rate = (
        total_answer_matches / total_questions
        if total_questions
        else 0
    )

    # Save all generated answers and evaluation information.
    output = {
        "total_questions": total_questions,
        "page_hit_rate": page_hit_rate,
        "answer_match_rate": answer_match_rate,
        "hallucination_pass_rate": hallucination_pass_rate,
        "category_stats": category_stats,
        "results": results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Print the final baseline summary.
    print("\n" + "=" * 50)
    print("BASELINE EVALUATION")
    print("=" * 50)

    print(f"Total questions: {total_questions}")
    print(
        f"Page hit rate: "
        f"{page_hit_rate:.2%}"
    )
    print(
        f"Answer match rate: "
        f"{answer_match_rate:.2%}"
    )
    print(
        f"Hallucination check pass rate: "
        f"{hallucination_pass_rate:.2%}"
    )

    print("\nBy category:")

    for category, stats in category_stats.items():
        category_page_rate = (
            stats["page_hits"] / stats["total"]
        )

        category_answer_rate = (
            stats["answer_matches"] / stats["total"]
        )

        category_hallucination_rate = (
            stats["hallucination_passes"] / stats["total"]
        )

        print(f"\n{category}")
        print(f"  Questions: {stats['total']}")
        print(
            f"  Page hit rate: "
            f"{category_page_rate:.2%}"
        )
        print(
            f"  Answer match rate: "
            f"{category_answer_rate:.2%}"
        )
        print(
            f"  Hallucination check pass rate: "
            f"{category_hallucination_rate:.2%}"
        )

    print(f"\nDetailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()