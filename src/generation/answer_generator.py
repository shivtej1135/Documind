from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import re

from vision.image_router import ImageRouter
from vision.vision_answer_generator import VisionAnswerGenerator


# Load environment variables
load_dotenv()


FIGURE_TABLE_RE = re.compile(
    r"[Ff]igure\s+\d+|[Tt]able\s+\d+"
)


class AnswerGenerator:
    """
    Generates answers using retrieved chunks.
    Uses vision model when a figure/table is detected.
    """

    MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"

    def __init__(self):

        # Image routing
        self.image_router = ImageRouter()

        # Vision model
        self.vision_generator = VisionAnswerGenerator()

        # Text model
        self.llm = ChatOpenRouter(
            model=self.MODEL_NAME,
            temperature=0,
        )

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> dict:

        # -------------------------
        # Top retrieved chunk
        # -------------------------
        top_chunk = retrieved_chunks[0]

        print("\nTop chunk:")
        print(top_chunk["doc_id"])
        print(top_chunk["page_num"])
        print(top_chunk["chunk_type"])

        # -------------------------
        # Image routing
        # -------------------------
        needs_image = self.image_router.needs_image(
            question,
            top_chunk,
        )

        print(f"Needs image: {needs_image}")

        vision_answer = None

        if needs_image:

            match = FIGURE_TABLE_RE.search(question)

            if match:
                figure_name = match.group()

                image_info = (
                    self.image_router.find_figure_image(
                        figure_name
                    )
                )

            else:
                image_info = self.image_router.find_image(
                    top_chunk["doc_id"],
                    top_chunk["page_num"],
                )

            print("\nImage found:")
            print(image_info)

            if image_info is not None:

                vision_answer = (
                    self.vision_generator.answer_from_image(
                        question=question,
                        image_path=image_info["image_path"],
                        caption=image_info["caption"],
                    )
                )

                print("\nVision answer:")
                print(vision_answer)

        # -------------------------
        # Build context
        # -------------------------
        context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        # -------------------------
        # Add vision output to prompt
        # -------------------------
        if vision_answer:

            prompt = f"""
Answer the question using the retrieved text and image analysis.

Retrieved Context:
{context}

Image Analysis:
{vision_answer}

Question:
{question}

Give a concise factual answer.
"""

        else:

            prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

If the context does not contain enough information to answer
the question, say that the answer cannot be determined from
the provided context.
"""

        # -------------------------
        # Generate answer
        # -------------------------
        response = self.llm.invoke(prompt)

        answer = response.content

        # -------------------------
        # Build sources
        # -------------------------
        sources = []

        for chunk in retrieved_chunks:

            source = {
                "doc_id": chunk["doc_id"],
                "page_num": chunk["page_num"],
                "chunk_type": chunk["chunk_type"],
                "section_heading": chunk["section_heading"],
            }

            if source not in sources:
                sources.append(source)

        # -------------------------
        # Hallucination check
        # -------------------------
        unsupported_numbers = (
            self._find_unsupported_numbers(
                answer,
                context,
            )
        )

        if unsupported_numbers:

            print(
                "\nWARNING: Unsupported numbers found:"
            )

            for number in unsupported_numbers:
                print(f"- {number}")

        return {
            "answer": answer,
            "sources": sources,
            "hallucination_check": {
                "unsupported_numbers": unsupported_numbers,
                "passed": len(unsupported_numbers) == 0,
            },
        }

    def _find_unsupported_numbers(
        self,
        answer: str,
        context: str,
    ) -> list[str]:
        """
        Very simple hallucination checker.
        """

        numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            answer,
        )

        unsupported_numbers = []

        for number in numbers:

            if number not in context:
                unsupported_numbers.append(number)

        return list(
            dict.fromkeys(
                unsupported_numbers
            )
        )