from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class GenerateMailPromotionOutput(BaseModel):
    offer_description: str
    discount_percentage: float
    validity_date: Optional[datetime] = None
    coupon_code: str
    discount_value: float
    coupon_expiry: Optional[datetime] = None
    product_description: str
    product_launch_date: Optional[datetime] = None
    customer_preferences: str
    historical_purchases: str
