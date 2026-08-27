# tests/unit/test_groq_client.py
import pytest
from unittest.mock import patch, MagicMock
from src.groq_client import GroqEngine
from src.schema import ReviewInput, BatchABSAResponse


@patch("src.groq_client.Groq")
def test_analyze_reviews_batch_parses_json_schema(mock_groq_cls):
    """Tests GroqEngine JSON completion formatting and Pydantic validation."""
    mock_client = MagicMock()
    mock_groq_cls.return_value = mock_client

    # Realistic raw LLM JSON response payload
    raw_json = """{
        "reviews": [
            {
                "review_id": 1,
                "summary": "Great food",
                "aspects": [
                    {"aspect": "FOOD", "sentiment": "positive", "evidence": "맛있어요"}
                ]
            }
        ]
    }"""
    
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = raw_json
    mock_client.chat.completions.create.return_value = mock_completion

    with patch.dict("os.environ", {"GROQ_API_KEY": "mock_key"}):
        engine = GroqEngine()
        batch_input = [ReviewInput(id=1, text="맛있어요")]
        result = engine.analyze_reviews_batch(batch_input)

        assert isinstance(result, BatchABSAResponse)
        assert len(result.reviews) == 1
        assert result.reviews[0].review_id == 1
        assert result.reviews[0].aspects[0].aspect == "FOOD"