import json
import re


# Detect Figure 1, Figure 2, Table 1, etc.
FIGURE_TABLE_RE = re.compile(
    r"[Ff]igure\s+\d+|[Tt]able\s+\d+"
)


class ImageRouter:

    def __init__(self):

        # Load visual metadata once
        with open(
            "data/visual_elements.json",
            "r",
            encoding="utf-8",
        ) as file:
            self.visual_elements = json.load(file)

    def needs_image(
        self,
        question,
        retrieved_chunk,
    ):
        # Table chunks should use images
        if retrieved_chunk.get("chunk_type") == "table":
            return True

        # Check for Figure X / Table X in the question
        match = FIGURE_TABLE_RE.search(question)

        if match:
            print(
                f"Figure/Table detected: {match.group()}"
            )
            return True

        return False

    def find_image(
        self,
        doc_id,
        page_num,
    ):
        # Find image using document and page number
        for item in self.visual_elements:

            if (
                item["doc_id"] == doc_id
                and item["page_num"] == page_num
            ):
                return item

        return None

    def find_figure_image(
        self,
        figure_name,
    ):
        # Find image using caption text
        for item in self.visual_elements:

            caption = item.get("caption", "")

            if (
                figure_name.lower()
                in caption.lower()
            ):
                return item

        return None

    def extract_figure_name(
        self,
        text,
    ):
        match = FIGURE_TABLE_RE.search(text)

        if match:
            return match.group()

        return None