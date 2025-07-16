from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class SummarizeProductReviewsOutput(BaseModel):
    user_name: str
    product_id: str
    product_description: str
    product_specification: str
    brand: str
    customer_reviews: str
    customer_rating: float
    review_sentiment: str
    price: float
