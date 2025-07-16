from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CreateProductDescriptionOutput(BaseModel):
    product_id: str
    product_description: str
    product_specification: str
    product_category: str
    brand: str
    price: float
    customer_reviews: str
    customer_rating: float
