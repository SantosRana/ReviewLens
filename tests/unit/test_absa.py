# tests/unit/test_absa.py
from unittest.mock import MagicMock
from src.absa import ABSAOrchestrator
from src.schema import ReviewInput, ReviewABSA, AspectSentiment, BatchABSAResponse


def test_process_reviews_executes_pipeline_and_batching():
    """Executes real process_reviews, _chunk_list, and _aggregate_metrics across batches."""
    orchestrator = ABSAOrchestrator(batch_size=5)

    # Mock engine return payload for each batch
    mock_batch_response = BatchABSAResponse(
        reviews=[
            ReviewABSA(
                review_id=1,
                summary="Delicious food",
                aspects=[
                    AspectSentiment(aspect="FOOD", sentiment="positive", evidence="국물이 정말 얼큰함")
                ]
            )
        ]
    )

    orchestrator.engine.analyze_reviews_batch = MagicMock(return_value=mock_batch_response)

    # 11 reviews forces 3 batch calls (chunked at batch_size=5)
    reviews = [f"국물이 정말 얼큰함 리뷰 {i}" for i in range(11)]
    result = orchestrator.process_reviews(reviews)

    # Assert real pipeline execution assertions
    assert result["status"] == "success"
    assert result["total_reviews_analyzed"] == 11
    assert "FOOD" in result["aspect_scores"]
    assert result["aspect_scores"]["FOOD"]["score"] == 100.0
    assert orchestrator.engine.analyze_reviews_batch.call_count == 3


def test_process_reviews_handles_empty_input():
    orchestrator = ABSAOrchestrator()
    result = orchestrator.process_reviews([])
    assert result["status"] == "empty"
    assert result["total_reviews_analyzed"] == 0


def test_validator_verbatim_grounding_pass(validator):
    raw_input = ReviewInput(id=1, text="국물이 정말 얼큰하고 시원합니다.")
    parsed = ReviewABSA(
        review_id=1,
        summary="Good soup",
        aspects=[AspectSentiment(aspect="FOOD", sentiment="positive", evidence="국물이 정말 얼큰")]
    )
    validated = validator.validate_review(parsed, raw_input)
    assert len(validated.aspects) == 1


def test_validator_hallucination_filtered(validator):
    raw_input = ReviewInput(id=1, text="국물이 정말 얼큰하고 시원합니다.")
    parsed = ReviewABSA(
        review_id=1,
        summary="Fake quote",
        aspects=[AspectSentiment(aspect="SERVICE", sentiment="positive", evidence="직원이 매우 친절함")]
    )
    validated = validator.validate_review(parsed, raw_input)
    assert len(validated.aspects) == 0