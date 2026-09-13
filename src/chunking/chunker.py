import re


class DocumentChunker:
    """
    Converts the structured JSON produced by our Docling parser
    into retrieval-ready chunks.
    """

    # Target a practical chunk size before embedding.
    # This is a simple word-based approximation of the
    # 300-500 token target from the Day 2 specification.
    MAX_WORDS = 350

    # Very small, non-sentence fragments are usually PDF
    # furniture such as graph labels or stray extracted tokens.
    MIN_FRAGMENT_WORDS = 4

    # Author/affiliation lines before the first section heading
    # are usually short and do not end with punctuation.
    MAX_FRONT_MATTER_WORDS = 12

    # Docling labels that identify actual section headings.
    HEADING_LABELS = {
        "section_header",
        "heading",
    }

    # Content that should not become retrieval chunks.
    IGNORED_LABELS = {
        "page_header",
        "page_footer",
        "footnote",
    }

    # Labels representing normal textual content.
    TEXT_LABELS = {
        "paragraph",
        "text",
        "list_item",
    }

    # Used to remove author contact information and similar
    # metadata that is not useful for document retrieval.
    EMAIL_OR_URL_RE = re.compile(
        r"[\w.\-]+@[\w.\-]+|https?://\S+|www\.\S+"
    )

    # Used when identifying short front-matter lines.
    SENTENCE_END_RE = re.compile(r"[.!?]\s*$")

    def chunk_document(self, document: dict) -> list[dict]:
        """
        Convert one Day-1 structured JSON document into chunks.
        """
        doc_id = document["doc_id"]

        chunks = []
        chunk_id = 0

        # Stores the section heading currently being processed.
        current_section = None

        # Consecutive prose is accumulated before becoming a chunk.
        current_text = []
        current_text_page = None

        # True while processing title/author information before
        # the first real section heading.
        in_front_matter = False

        def add_text_chunk():
            """
            Convert accumulated prose into one or more chunks.
            """
            nonlocal chunk_id
            nonlocal current_text
            nonlocal current_text_page

            if not current_text:
                return

            text = " ".join(current_text).strip()
            page_num = current_text_page

            # Clear the accumulator before creating chunks.
            current_text = []
            current_text_page = None

            if not text:
                return

            # Split only when the accumulated text is too long.
            text_parts = self._split_long_text(text)

            for part in text_parts:
                chunk_id += 1

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "chunk_type": "text",
                        "section_heading": current_section,
                        "text": part,
                    }
                )

        # Process pages in the same order as the JSON.
        for page in document["pages"]:
            page_num = page["page_num"]

            for text_block in page["text_blocks"]:
                text = text_block["text"].strip()
                label = text_block["label"]

                if not text:
                    continue

                # Ignore PDF headers, footers and footnotes.
                if label in self.IGNORED_LABELS:
                    continue

                # Ignore email/URL metadata such as author contact
                # information.
                if self.EMAIL_OR_URL_RE.search(text):
                    continue

                # A section heading ends the previous accumulated
                # text and becomes the active section.
                if label in self.HEADING_LABELS:
                    add_text_chunk()

                    current_section = text
                    in_front_matter = False
                    continue

                # The document title is kept as its own chunk and
                # does not become the section heading for the authors.
                if label == "title":
                    add_text_chunk()

                    chunk_id += 1

                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "page_num": page_num,
                            "chunk_type": "text",
                            "section_heading": None,
                            "text": text,
                        }
                    )

                    current_section = None
                    in_front_matter = True
                    continue

                # Ignore likely author/affiliation lines between
                # the title and the first real section.
                if (
                    in_front_matter
                    and self._looks_like_front_matter_noise(text)
                ):
                    continue

                # Accumulate normal prose so related text becomes
                # one coherent retrieval chunk.
                if label in self.TEXT_LABELS:
                    if self._is_orphan_fragment(text):
                        continue

                    if not current_text:
                        current_text_page = page_num

                    current_text.append(text)
                    continue

                # Preserve other meaningful text types as their own
                # chunks instead of silently discarding them.
                add_text_chunk()

                chunk_id += 1

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "chunk_type": "text",
                        "section_heading": current_section,
                        "text": text,
                    }
                )

        # Flush any prose remaining after the last page.
        add_text_chunk()

        # Tables are kept intact and stored as separate retrieval chunks.
        for table in document["tables"]:
            rows = table.get("rows", [])

            if not rows:
                continue

            table_text = self._table_to_text(rows)

            if not table_text.strip():
                continue

            chunk_id += 1

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "page_num": table.get("page_num"),
                    "chunk_type": "table",
                    "section_heading": None,
                    "text": table_text,
                }
            )

        return chunks

    def _looks_like_front_matter_noise(self, text: str) -> bool:
        """
        Identify short, unpunctuated author/affiliation lines
        appearing before the first section heading.
        """
        word_count = len(text.split())

        return (
            word_count <= self.MAX_FRONT_MATTER_WORDS
            and not self.SENTENCE_END_RE.search(text)
        )

    def _is_orphan_fragment(self, text: str) -> bool:
        """
        Remove very short fragments that are unlikely to contain
        useful semantic information.
        """
        word_count = len(text.split())

        return (
            word_count < self.MIN_FRAGMENT_WORDS
            and not self.SENTENCE_END_RE.search(text)
        )

    def _split_long_text(self, text: str) -> list[str]:
        """
        Keep normal prose intact.

        Split unusually long prose at sentence boundaries instead
        of cutting it at an arbitrary character/word position.
        """
        if len(text.split()) <= self.MAX_WORDS:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_word_count = len(sentence.split())

            # Start a new chunk before exceeding the word limit.
            if (
                current_sentences
                and current_word_count + sentence_word_count > self.MAX_WORDS
            ):
                chunks.append(" ".join(current_sentences))

                current_sentences = []
                current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += sentence_word_count

        # Add the final group of sentences.
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _table_to_text(self, rows: list[list[str]]) -> str:
        """
        Convert structured table rows into readable text while
        keeping the entire table together as one chunk.
        """
        return "\n".join(
            " | ".join(str(cell).strip() for cell in row)
            for row in rows
        )