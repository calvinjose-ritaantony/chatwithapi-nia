from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ProductInformationOutput(BaseModel):
    user_name: str
    product_id: str
    product_description: str
    brand: str
    price: float
    product_category: str
    delivery_date: Optional[datetime] = None
    customer_reviews: str
    customer_rating: float
