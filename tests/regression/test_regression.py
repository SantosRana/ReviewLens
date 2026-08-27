# tests/regression/test_regression.py

def test_regression_short_korean_phrases_preserved(validator):
    """Ensures short 2-3 char Korean evidence quotes like '찐맛집' are not dropped as sub-length quotes."""
    from src.schema import ReviewInput, ReviewABSA, AspectSentiment
    
    raw = ReviewInput(id=1, text="여기 삼청동 찐맛집 맞네요.")
    parsed = ReviewABSA(
        review_id=1, summary="ok",
        aspects=[AspectSentiment(aspect="FOOD", sentiment="positive", evidence="찐맛집")]
    )
    
    validated = validator.validate_review(parsed, raw)
    assert len(validated.aspects) == 1
    assert validated.aspects[0].evidence == "찐맛집"