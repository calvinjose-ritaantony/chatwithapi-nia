from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocSearchOutput(BaseModel):
    title: str
    chunk: str
