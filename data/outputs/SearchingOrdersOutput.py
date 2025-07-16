from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class SearchingOrdersOutput(BaseModel):
    user_name: str
    product_id: str
    product_description: str
    brand: str
    order_id: str
    order_date: datetime
    order_total: float
    status: str
    delivery_date: Optional[datetime] = None
    shipping_address: str
    payment_method: str
    payment_status: str