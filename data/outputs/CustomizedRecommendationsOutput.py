from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CustomizedRecommendationsOutput(BaseModel):
    user_name: str
    order_id: str
    product_id: str
    product_description: str
    brand: str
    price: float
    order_date: Optional[datetime] = None
    order_total: float
    qty: float
    country: str
