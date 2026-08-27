# tests/integration/test_pipeline.py
from unittest.mock import MagicMock


def test_end_to_end_search_flow_and_state(mock_agent):
    """Verifies search execution populates both cache and session state pointers for follow-ups."""
    mock_agent.naver_search.fetch_reviews = MagicMock(return_value=["맛있어요", "친절해요"])
    mock_agent.orchestrator.process_reviews = MagicMock(return_value={
        "status": "success",
        "overall_score": 4.8,
        "aspect_scores": {},
        "pros": ["맛있어요"],
        "cons": [],
        "pipeline_metrics": {"retrieved": 2, "analyzed": 2}
    })

    res = mock_agent.search_and_analyze_restaurant("삼청동수제비")

    # 1. Output dict assertions
    assert res["status"] == "success"
    assert res["restaurant_name"] == "삼청동수제비"

    # 2. Session state persistence assertions (required for follow-up skills)
    assert "삼청동수제비" in mock_agent.analysis_cache
    assert mock_agent.current_restaurant == "삼청동수제비"
    assert mock_agent.current_analysis is not None