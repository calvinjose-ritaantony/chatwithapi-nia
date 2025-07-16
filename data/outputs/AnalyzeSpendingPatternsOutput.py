from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class AnalyzeSpendingPatternsOutput(BaseModel):
    user_name: str
    product_id: str
    product_description: str
    product_category: str
    brand: str
    price: float
    order_id: str
    order_date: Optional[datetime] = None
    order_total: float
    payment_method: str
    payment_status: str
