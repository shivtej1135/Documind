from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


class DoclingParser:
    def __init__(self):
        # Enable image generation for figures and tables.
        pipeline_options = PdfPipelineOptions()

        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.generate_page_images = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def parse(self, pdf_path: str | Path) -> dict:
        pdf_path = Path(pdf_path)

        # Convert the PDF into Docling's structured document object.
        result = self.converter.convert(str(pdf_path))
        document = result.document

        parsed_doc = {
            "doc_id": pdf_path.stem,
            "pages": [],
            "tables": [],
            "pictures": [],
        }

        # -------------------------
        # TEXT BLOCKS
        # -------------------------
        pages_dict = {}

        for text_item in document.texts:
            if not text_item.prov:
                continue

            # Ignore page headers and footers because they are not
            # useful document content for RAG.
            label = text_item.label.value

            if label in {"page_header", "page_footer"}:
                continue

            page_num = text_item.prov[0].page_no

            if page_num not in pages_dict:
                pages_dict[page_num] = []

            # Keep the label so the chunker can distinguish
            # headings from normal paragraphs.
            pages_dict[page_num].append(
                {
                    "text": text_item.text,
                    "label": label,
                }
            )

        # Save pages in page-number order.
        for page_num in sorted(pages_dict):
            parsed_doc["pages"].append(
                {
                    "page_num": page_num,
                    "text_blocks": pages_dict[page_num],
                }
            )

        # -------------------------
        # TABLES
        # -------------------------
        for table_index, table in enumerate(document.tables, start=1):
            page_num = None

            if table.prov:
                page_num = table.prov[0].page_no

            # Convert Docling's structured table into rows and columns.
            dataframe = table.export_to_dataframe(doc=document)

            rows = [dataframe.columns.tolist()]
            rows.extend(dataframe.fillna("").astype(str).values.tolist())

            parsed_doc["tables"].append(
                {
                    "table_id": f"table_{table_index}",
                    "page_num": page_num,
                    "caption": table.caption_text(document),
                    "rows": rows,
                }
            )

        # -------------------------
        # PICTURES
        # -------------------------
        figure_dir = (
            pdf_path.parent.parent
            / "output"
            / pdf_path.stem
            / "figures"
        )

        figure_dir.mkdir(parents=True, exist_ok=True)

        for picture_index, picture in enumerate(
            document.pictures,
            start=1,
        ):
            page_num = None

            if picture.prov:
                page_num = picture.prov[0].page_no

            image_path = None

            # Ask Docling for the actual cropped picture image.
            image = picture.get_image(document)

            if image is not None:
                image_file = figure_dir / f"figure_{picture_index}.png"
                image.save(image_file)
                image_path = str(image_file)

            parsed_doc["pictures"].append(
                {
                    "picture_id": f"figure_{picture_index}",
                    "page_num": page_num,
                    "caption": picture.caption_text(document),
                    "image_path": image_path,
                }
            )

        return parsed_doc