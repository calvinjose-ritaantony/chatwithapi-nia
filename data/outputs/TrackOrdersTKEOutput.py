from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TrackOrdersTKEOutput(BaseModel):
    opportunity: str
    Product: str
    orderQuantity: float
    status: str
    createDate: Optional[datetime] = None
    organization: str
    solutionDescription: str
    totalCustomerSalePrice: float
    yourReferenceId: str
    dealerNumber: str
    solutionOrgLabel: str
    organizationType: str
    noOfDaysToDrawingsExpire: float
    solutionId: str
    orderClass: str
