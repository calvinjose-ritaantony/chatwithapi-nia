from pydantic import BaseModel,Field
from typing import List, Literal, Optional, Union
from datetime import datetime

# class TextBlock(BaseModel):
#     block_type: Literal["text"]
#     title: str
#     content: str

# class TableBlock(BaseModel):
#     block_type: Literal["table"]
#     title: str
#     headers: List[str]
#     rows: List[List[str]]

# class ChartBlock(BaseModel):
#     block_type: Literal["chart"]
#     chart_type: Literal["bar", "line", "pie"]
#     title: str
#     x: List[str]
#     y: List[float]

# class MailBlock(BaseModel):
#     block_type: Literal["mail"]
#     title: str
#     recipient: str
#     subject: str
#     body: str

class TextBlock(BaseModel):
    """
    Represents a simple text block.  
    Use this only if plain text is needed.
    Fields are optional.
    """
    block_type: Literal["text"]
    title: Optional[str] = Field(
        default=None,
        description="Optional heading for the text block."
    )
    content: Optional[str] = Field(
        default=None,
        description="The main text content. Leave empty if not needed."
    )


class TableBlock(BaseModel):
    """
    Represents a table with headers and rows.
    All fields are optional to allow flexibility.
    """
    block_type: Literal["table"]
    title: Optional[str] = Field(None, description="Optional title for the table.")
    headers: Optional[List[str]] = Field(None, description="List of column headers.")
    rows: Optional[List[List[str]]] = Field(None, description="List of table rows.")
    # title: Optional[str] = Field(
    #     default=None,
    #     description="Optional title for the table."
    # )
    # headers: Optional[List[str]] = Field(
    #     default=None,
    #     description="List of column headers. May be skipped if not needed."
    # )
    # rows: Optional[List[List[str]]] = Field(
    #     default=None,
    #     description="List of rows, where each row is a list of string values."
    # )


class ChartBlock(BaseModel):
    """
    Represents a chart with a given type and data points.
    Supports bar, line, and pie charts.
    """
    block_type: Literal["chart"]
    chart_type: Literal["bar", "line", "pie"]
    title: Optional[str] = Field(
        default=None,
        description="Optional title for the chart."
    )
    x: Optional[List[str]] = Field(
        default=None,
        description="X-axis labels or categories."
    )
    y: Optional[List[float]] = Field(
        default=None,
        description="Y-axis values corresponding to the X-axis labels."
    )


class MailBlock(BaseModel):
    """
    Represents a mail block with recipient, subject, and body.
    Used when sending or previewing email content.
    """
    block_type: Literal["mail"]
    title: Optional[str] = Field(
        default=None,
        description="Optional title for the mail block."
    )
    recipient: Optional[str] = Field(
        default=None,
        description="Email address of the recipient."
    )
    subject: Optional[str] = Field(
        default=None,
        description="Subject line of the email."
    )
    body: Optional[str] = Field(
        default=None,
        description="Main body content of the email."
    )

Block = Union[TextBlock, TableBlock, ChartBlock]

class StructuredSpendingAnalysis(BaseModel):
    type: Literal["structured_response"]
    title: str
    description: str
    blocks: List[Block]
    closure: str
    follow_up_questions: Optional[List[str]] = None  

    