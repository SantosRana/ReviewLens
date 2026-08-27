# tests/conftest.py
import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.guardrails import SecurityGuardrails
from src.validator import ABSAOutputValidator
from src.schema import ReviewInput


@pytest.fixture
def fixtures_dir():
    """Returns the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_reviews(fixtures_dir):
    """Loads review data fixtures."""
    with open(fixtures_dir / "test_reviews.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def test_queries(fixtures_dir):
    """Loads query fixtures."""
    with open(fixtures_dir / "test_queries.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def guardrails():
    """Provides a fresh SecurityGuardrails instance."""
    return SecurityGuardrails(max_input_length=1000)


@pytest.fixture
def validator():
    """Provides a fresh ABSAOutputValidator instance."""
    return ABSAOutputValidator()


@pytest.fixture
def sample_review_input():
    """Provides a canonical ReviewInput object."""
    return ReviewInput(
        id=1,
        text="삼청동수제비는 뜨끈한 국물이 일품이고 김치가 정말 맛있음. 다만 주차장이 따로 없어서 불편했음."
    )


@pytest.fixture
def mock_agent():
    """
    Initializes ReviewLensAgent with mocked Groq SDK and Naver API.
    Exposes mock_agent.client.chat.completions.create for assertion checks.
    """
    with patch.dict("os.environ", {"GROQ_API_KEY": "mock_test_key"}):
        with patch("src.agent.Groq") as mock_groq_cls:
            with patch("src.agent.NaverBlogSearch"):
                # Setup mock Groq SDK client
                mock_client = MagicMock()
                mock_groq_cls.return_value = mock_client

                from src.agent import ReviewLensAgent
                agent_instance = ReviewLensAgent()
                return agent_instance