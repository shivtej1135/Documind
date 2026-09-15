from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
import re

from vision.image_router import ImageRouter
from vision.vision_answer_generator import VisionAnswerGenerator


load_dotenv()


FIGURE_TABLE_RE = re.compile(
    r"[Ff]igure\s+\d+|[Tt]able\s+\d+"
)


class AnswerGenerator:
    """
    Generates the final answer using retrieved text.

    If a figure or table is relevant, the vision model is also used.
    The final response always follows the same unified schema.
    """

    MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"

    def __init__(self):
        self.image_router = ImageRouter()
        self.vision_generator = VisionAnswerGenerator()

        self.llm = ChatOpenRouter(
            model=self.MODEL_NAME,
            temperature=0,
        )

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
    ) -> dict:

        # The highest-ranked retrieved chunk is used
        # to decide whether the question needs an image.
        top_chunk = retrieved_chunks[0]

        print("\nTop chunk:")
        print(top_chunk["doc_id"])
        print(top_chunk["page_num"])
        print(top_chunk["chunk_type"])

        needs_image = self.image_router.needs_image(
            question,
            top_chunk,
        )

        print(f"Needs image: {needs_image}")

        vision_answer = None
        image_info = None

        # ---------------------------------------------------------
        # IMAGE / TABLE LOOKUP
        # ---------------------------------------------------------
        if needs_image:

            # Check whether the question explicitly mentions
            # a Figure X or Table X.
            match = FIGURE_TABLE_RE.search(question)

            if match:
                visual_name = match.group()

                # -------------------------------------------------
                # FIGURE LOOKUP
                # -------------------------------------------------
                if visual_name.lower().startswith("figure"):

                    checked_docs = set()

                    # Check retrieved documents in ranking order.
                    # This is important because figure numbers are
                    # not globally unique across documents.
                    for chunk in retrieved_chunks:

                        doc_id = chunk["doc_id"]

                        if doc_id in checked_docs:
                            continue

                        checked_docs.add(doc_id)

                        image_info = self.image_router.find_figure_image(
                            doc_id,
                            visual_name,
                        )

                        if image_info is not None:
                            break

                # -------------------------------------------------
                # TABLE LOOKUP
                # -------------------------------------------------
                elif visual_name.lower().startswith("table"):

                    checked_docs = set()

                    for chunk in retrieved_chunks:

                        doc_id = chunk["doc_id"]

                        if doc_id in checked_docs:
                            continue

                        checked_docs.add(doc_id)

                        image_info = self.image_router.find_table_image(
                            doc_id,
                            visual_name,
                        )

                        if image_info is not None:
                            break

            # No explicit Figure/Table number.
            # In this case use the visual element on the
            # retrieved chunk's page.
            else:
                image_info = self.image_router.find_image(
                    top_chunk["doc_id"],
                    top_chunk["page_num"],
                )

            print("\nImage found:")
            print(image_info)

            # Send the actual image to the vision model.
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

        # ---------------------------------------------------------
        # TEXT CONTEXT
        # ---------------------------------------------------------

        context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        # ---------------------------------------------------------
        # FINAL ANSWER GENERATION
        # ---------------------------------------------------------

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

        response = self.llm.invoke(prompt)

        answer = response.content

        # ---------------------------------------------------------
        # UNIFIED SOURCE SCHEMA
        # ---------------------------------------------------------

        sources = []

        # If an actual image was used, add that image as the
        # primary source.
        if vision_answer and image_info is not None:

            image_source = {
                "doc_id": image_info["doc_id"],
                "page_num": image_info["page_num"],
                "chunk_type": image_info["type"],
                "section_heading": None,
                "image_path": image_info["image_path"],
                "caption": image_info["caption"],
            }

            sources.append(image_source)

        # Add retrieved text/table chunks as supporting sources.
        for chunk in retrieved_chunks:

            source = {
                "doc_id": chunk["doc_id"],
                "page_num": chunk["page_num"],
                "chunk_type": chunk["chunk_type"],
                "section_heading": chunk["section_heading"],
                "image_path": None,
                "caption": None,
            }

            # Prevent duplicate sources.
            if source not in sources:
                sources.append(source)

        # ---------------------------------------------------------
        # SOURCE TYPE
        # ---------------------------------------------------------

        if vision_answer:
            source_type = "image"
        else:
            source_type = "text"

        # ---------------------------------------------------------
        # HALLUCINATION CHECK
        # ---------------------------------------------------------

        support_context = context

        if vision_answer:
            support_context += "\n\n" + vision_answer

        unsupported_numbers = self._find_unsupported_numbers(
            answer,
            support_context,
        )

        hallucination_flag = len(unsupported_numbers) > 0

        if unsupported_numbers:

            print("\nWARNING: Unsupported numbers found:")

            for number in unsupported_numbers:
                print(f"- {number}")

        # ---------------------------------------------------------
        # FINAL UNIFIED RESPONSE
        # ---------------------------------------------------------

        return {
            "question": question,
            "answer": answer,
            "source_type": source_type,
            "sources": sources,
            "hallucination_flag": hallucination_flag,
        }

    def _find_unsupported_numbers(
        self,
        answer: str,
        context: str,
    ) -> list[str]:
        """
        Finds numbers appearing in the answer that are not
        present in the supporting context.
        """

        numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            answer,
        )

        unsupported_numbers = []

        for number in numbers:

            if number not in context:
                unsupported_numbers.append(number)

        # Remove duplicates while preserving order.
        return list(dict.fromkeys(unsupported_numbers))