from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ReviewBytesOutput(BaseModel):
    title: str
    review: str
    price: float
    features: str
    rating: float
