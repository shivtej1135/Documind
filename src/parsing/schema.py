from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class TextBlock:
    id: str
    page_num: int
    text: str
    section_title: Optional[str] = None


@dataclass
class TableData:
    id: str
    page_num: int
    caption: str
    rows: List[List[str]]


@dataclass
class FigureData:
    id: str
    page_num: int
    caption: str
    image_path: str


@dataclass
class PageData:
    page_num: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    figures: List[FigureData] = field(default_factory=list)


@dataclass
class DocumentData:
    doc_id: str
    title: str
    pages: List[PageData]

    def to_dict(self):
        return asdict(self)