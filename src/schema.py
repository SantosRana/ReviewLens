from pydantic import BaseModel, Field
from typing import List, Literal

class AspectSentiment(BaseModel):
    aspect: Literal["FOOD", "PRICE", "SERVICE", "AMBIENCE"] = Field(
        description="The target aspect category evaluated in the review text"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The detected sentiment polarity for this aspect"
    )
    evidence: str = Field(
        description=(
            "Exact Korean substring from the review that directly supports "
            "the assigned aspect and sentiment. Do not paraphrase or invent evidence."
        )
    )

class ReviewABSA(BaseModel):
    review_id: int = Field(
        description="1-based index of the review within the current batch"
    )
    aspects: List[AspectSentiment] = Field(
        description="Collection of aspect-sentiment pairs extracted from the review"
    )
    summary: str = Field(
        default="",
        description=(
            "Single concise sentence summarizing the key aspect-sentiment findings "
            "of this review in Korean or English."
        )
    )

class BatchABSAResponse(BaseModel):
    reviews: List[ReviewABSA] = Field(
        description="Structured ABSA extraction results for all reviews in the submitted batch"
    )
    
class ReviewInput(BaseModel):
    id: int = Field(ge=1, description="Immutable 1-based review identifier")
    text: str = Field(min_length=1, description="Raw review snippet text")