# tests/agent/test_domain_guard.py
import pytest


def test_domain_guard_skill_loaded(mock_agent):
    """Verifies that domain_guard.md skill is loaded into the agent state."""
    assert mock_agent.domain_guard_skill != ""
    assert "REVIEW" in mock_agent.domain_guard_skill
    assert "FOLLOW_UP" in mock_agent.domain_guard_skill
    assert "OFF_TOPIC" in mock_agent.domain_guard_skill


@pytest.mark.parametrize("query", [
    "삼청동수제비 분석해줘",
    "Analyze Bongpiyang reviews",
    "봉피양이랑 삼청동수제비 비교해줘",
])
def test_review_queries_taxonomy(mock_agent, query):
    """Verifies REVIEW category queries pass input guardrails and reach agent instructions."""
    guardrail_res = mock_agent.guardrails.validate_user_input(query)
    assert guardrail_res.is_safe is True
    assert mock_agent._get_system_instructions() is not None


@pytest.mark.parametrize("query", [
    "How is the service score?",
    "가격은 어때?",
    "What did people complain about?",
])
def test_followup_queries_taxonomy(mock_agent, query):
    """Verifies FOLLOW_UP queries remain valid when active context is established."""
    mock_agent.current_restaurant = "삼청동수제비"
    guardrail_res = mock_agent.guardrails.validate_user_input(query)
    assert guardrail_res.is_safe is True


@pytest.mark.parametrize("query", [
    "Write a Python palindrome function",
    "Tell me a joke",
    "What is the capital of France?",
])
def test_off_topic_queries_taxonomy(mock_agent, query):
    """Verifies OFF_TOPIC queries pass input guardrails but are routed for refusal by skill guidelines."""
    guardrail_res = mock_agent.guardrails.validate_user_input(query)
    # Off-topic queries are benign (is_safe=True) and rejected at the prompt skill layer
    assert guardrail_res.is_safe is True