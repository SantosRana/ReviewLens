# tests/unit/test_naver.py
import pytest
from unittest.mock import patch, MagicMock
from src.naver import NaverBlogSearch


@patch("requests.get")
def test_fetch_reviews_parses_and_cleans_html(mock_get):
    """Tests Naver API JSON parsing and HTML tag removal."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [
            {
                "title": "<b>삼청동수제비</b> 방문 후기",
                "description": "<b>국물이 깊고 맛있어요</b> 직원분들도 정말 친절했습니다."
            }
        ]
    }
    mock_get.return_value = mock_response

    search = NaverBlogSearch()
    reviews = search.fetch_reviews("삼청동수제비", display_count=1)

    assert len(reviews) == 1
    assert "<b>" not in reviews[0]
    assert "국물이 깊고 맛있어요 직원분들도 정말 친절했습니다." in reviews[0]


@patch("requests.get")
def test_fetch_reviews_handles_api_failure(mock_get):
    """Verifies graceful empty list return on HTTP error."""
    mock_get.side_effect = Exception("API connection timeout")
    search = NaverBlogSearch()
    reviews = search.fetch_reviews("삼청동수제비")
    assert reviews == []