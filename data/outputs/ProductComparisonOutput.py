from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ProductComparisonOutput(BaseModel):
    user_name: str
    product_id: str
    product_description: str
    brand: str
    price: float
    product_specification: str
    qty: float
