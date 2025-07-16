from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CustomerComplaintsOutput(BaseModel):
    user_name: str
    order_id: str
    product_id: str
    product_description: str
    order_date: Optional[datetime] = None
    order_total: float
    status: str
    delivery_date: Optional[datetime] = None
    customer_name: str
    customer_reviews: str
    return_policy: str
    payment_status: str
