# tests/unit/test_guardrails.py

import pytest
from src.guardrails import GuardrailReason, RiskLevel


# ------------------------------------------------------------------
# Boundary 1: Direct User Queries
# ------------------------------------------------------------------

def test_boundary1_valid_query_passes(guardrails):
    result = guardrails.validate_user_input("삼청동수제비 분석해줘")

    assert result.is_safe is True
    assert result.reason == GuardrailReason.PASSED


def test_boundary1_injection_blocked(guardrails, test_queries):
    for prompt in test_queries["malicious_injection"]:
        result = guardrails.validate_user_input(prompt)

        assert result.is_safe is False
        assert result.risk == RiskLevel.HIGH


def test_boundary1_empty_input_blocked(guardrails):
    result = guardrails.validate_user_input("")

    assert result.is_safe is False
    assert result.reason == GuardrailReason.EMPTY_INPUT


def test_boundary1_oversized_input_blocked(guardrails):
    oversized = "맛집 " * 400

    result = guardrails.validate_user_input(oversized)

    assert result.is_safe is False
    assert result.reason == GuardrailReason.INPUT_TOO_LONG


# ------------------------------------------------------------------
# Boundary 2: External / Custom Review Data
# ------------------------------------------------------------------

def test_boundary2_injection_reviews_rejected(guardrails, test_reviews):
    """Prompt injections inside review bodies must be rejected."""
    for raw_review in test_reviews["injection_reviews"]:
        sanitized = guardrails.validate_review_data(raw_review)

        assert sanitized is None


def test_boundary2_valid_review_sanitized(guardrails, test_reviews):
    """Legitimate reviews with control chars must be sanitized."""
    for raw_review in test_reviews["dirty_reviews"]:
        sanitized = guardrails.validate_review_data(raw_review)

        assert sanitized is not None
        assert "\x00" not in sanitized
        assert "\x08" not in sanitized
        assert "맛있어요." in sanitized
        assert "직원도 친절합니다." in sanitized


# ------------------------------------------------------------------
# Display Sanitization
# ------------------------------------------------------------------

def test_sanitize_for_display(guardrails):
    """System-style display tags must be removed."""
    raw = "[SYSTEM] hello [USER] world"

    cleaned = guardrails.sanitize_for_display(raw)

    assert "[SYSTEM]" not in cleaned
    assert "[USER]" not in cleaned
    assert "hello" in cleaned
    assert "world" in cleaned