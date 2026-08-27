# tests/unit/test_schema.py
import pytest
from pydantic import ValidationError
from src.schema import AspectSentiment, ReviewABSA


def test_aspect_sentiment_schema():
    valid = AspectSentiment(aspect="FOOD", sentiment="positive", evidence="맛있음")
    assert valid.aspect == "FOOD"

    with pytest.raises(ValidationError):
        AspectSentiment(aspect="INVALID_CATEGORY", sentiment="positive", evidence="맛있음")