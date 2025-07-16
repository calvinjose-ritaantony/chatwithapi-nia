from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class SeasonalSalesOutput(BaseModel):
    sales_period: str
    total_sales: float
    quantity_sold: float
    offer_description: str
    discount_percentage: float
    sale_date: Optional[datetime] = None
    sales_performance: str
    customer_behavior: str
