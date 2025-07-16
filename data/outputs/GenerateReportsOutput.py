from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class GenerateReportsOutput(BaseModel):
    user_name: str
    product_id: str
    order_id: str
    product_description: str
    price: float
    order_total: float
    qty: float
    order_date: Optional[datetime] = None
    customer_rating: float
    product_category: str
    delivery_date: Optional[datetime] = None
    customer_reviews: str
