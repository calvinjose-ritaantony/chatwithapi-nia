from pydantic import BaseModel
from typing import List, Literal, Optional, Union
from datetime import datetime

class TextBlock(BaseModel):
    block_type: Literal["text"]
    title: str
    content: str

class TableBlock(BaseModel):
    block_type: Literal["table"]
    title: str
    headers: List[str]
    rows: List[List[str]]

class ChartBlock(BaseModel):
    block_type: Literal["chart"]
    chart_type: Literal["bar", "line", "pie"]
    title: str
    x: List[str]
    y: List[float]

Block = Union[TextBlock, TableBlock, ChartBlock]

class StructuredSpendingAnalysis(BaseModel):
    type: Literal["structured_response"]
    title: str
    description: str
    blocks: List[Block]
    closure: str
    