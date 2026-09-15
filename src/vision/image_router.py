import json
import re


FIGURE_TABLE_RE = re.compile(
    r"[Ff]igure\s+\d+|[Tt]able\s+\d+"
)


class ImageRouter:
    def __init__(self):
        # Load metadata for all extracted figures and tables.
        with open(
            "data/visual_elements.json",
            "r",
            encoding="utf-8",
        ) as file:
            self.visual_elements = json.load(file)

    def needs_image(self, question, retrieved_chunk):
        # A retrieved table chunk should use the table image.
        if retrieved_chunk.get("chunk_type") == "table":
            return True

        # Explicit Figure X / Table X questions should use an image.
        match = FIGURE_TABLE_RE.search(question)

        if match:
            print(f"Figure/Table detected: {match.group()}")
            return True

        return False

    def find_image(self, doc_id, page_num):
        # Find any visual element on the given document page.
        for item in self.visual_elements:
            if (
                item["doc_id"] == doc_id
                and item["page_num"] == page_num
            ):
                return item

        return None

    def find_figure_image(self, doc_id, figure_name):
        # Find a specific figure using its caption.
        # Restrict the search to the correct document.
        for item in self.visual_elements:
            if item["doc_id"] != doc_id:
                continue

            if item["type"] != "figure":
                continue

            caption = item.get("caption", "")

            # Match the complete figure name.
            # This prevents Figure 1 from matching Figure 10.
            if re.search(
                rf"\b{re.escape(figure_name)}\b",
                caption,
                re.IGNORECASE,
            ):
                return item

        return None

    def find_table_image(self, doc_id, table_name):
        # Extract the table number from "Table X".
        match = re.search(
            r"[Tt]able\s+(\d+)",
            table_name,
        )

        if not match:
            return None

        table_number = match.group(1)

        # Search only inside the requested document.
        for item in self.visual_elements:
            if item["doc_id"] != doc_id:
                continue

            if item["type"] != "table":
                continue

            # Our generated metadata uses filenames such as:
            # Denseresultstable_p5_table2.png
            image_path = item.get("image_path", "")

            if re.search(
                rf"_table{table_number}\.png$",
                image_path,
                re.IGNORECASE,
            ):
                return item

        return None

    def extract_figure_name(self, text):
        match = FIGURE_TABLE_RE.search(text)

        if match:
            return match.group()

        return None