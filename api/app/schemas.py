from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class AskerInfo(BaseModel):
    """
    All fields optional. If omitted entirely, the asker is treated the same way
    a deleted/anonymized Stack Overflow account was treated during training:
    missing-indicator flags set to 1, values filled with the -1 sentinel.
    """
    reputation: Optional[int] = Field(None, ge=0)
    upvote_count: Optional[int] = Field(None, ge=0)
    downvote_count: Optional[int] = Field(None, ge=0)
    account_created: Optional[datetime] = None


class QuestionInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    body: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list, max_items=10)
    asker: Optional[AskerInfo] = None
    posted_at: Optional[datetime] = Field(
        None, description="Defaults to current UTC time if omitted."
    )


class TopFactor(BaseModel):
    feature: str
    plain_language: str
    direction: str  # "increases" or "decreases"
    impact: float


class TimeEstimate(BaseModel):
    likely_time: str
    range_low: str
    range_high: str
    summary: str


class PredictionResponse(BaseModel):
    probability_answered: float
    will_likely_be_answered: bool
    decision_threshold: float
    expected_time_to_answer: TimeEstimate
    top_factors: List[TopFactor]
    summary: str
