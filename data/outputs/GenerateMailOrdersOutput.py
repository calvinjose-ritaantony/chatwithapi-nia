from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class GenerateMailOrdersOutput(BaseModel):
    user_name: str
    email: str
    phone_number: str
    order_id: str
    product_id: str
    product_description: str
    brand: str
    price: float
    order_total: float
    order_date: Optional[datetime] = None
