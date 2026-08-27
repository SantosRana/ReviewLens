import unicodedata
import logging
from typing import List, Set, Tuple

from src.schema import ReviewABSA, ReviewInput, AspectSentiment

logger = logging.getLogger(__name__)

VALID_ASPECTS = {
    "FOOD",
    "PRICE",
    "SERVICE",
    "AMBIENCE",
}

VALID_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
}


class ABSAOutputValidator:
    """
    Deterministic post-LLM validator.

    Validates:
    1. Aspect domain
    2. Sentiment domain
    3. Evidence presence
    4. Minimum evidence length
    5. Evidence grounding against canonical review text
    6. Duplicate aspect/evidence removal

    ReviewInput is the canonical source of truth for grounding.
    """

    def validate_review(
        self,
        parsed_review: ReviewABSA,
        raw_review: ReviewInput,
    ) -> ReviewABSA:

        raw_text_norm = unicodedata.normalize(
            "NFKC",
            raw_review.text
        )

        validated_aspects: List[AspectSentiment] = []
        seen: Set[Tuple[str, str, str]] = set()

        for aspect_item in parsed_review.aspects:

            # --------------------------------------------------
            # 1. Domain validation
            # --------------------------------------------------
            if aspect_item.aspect not in VALID_ASPECTS:
                logger.warning("[Validator] Invalid aspect dropped ""(ID=%s): %r", raw_review.id, aspect_item.aspect)
                continue

            if aspect_item.sentiment not in VALID_SENTIMENTS:
                logger.warning("[Validator] Invalid sentiment dropped ""(ID=%s): %r", raw_review.id, aspect_item.sentiment)
                continue

            # --------------------------------------------------
            # 2. Evidence presence
            # --------------------------------------------------
            if not aspect_item.evidence:
                logger.warning("[Validator] Empty evidence dropped (ID=%s)", raw_review.id)
                continue

            evidence_norm = unicodedata.normalize(
                "NFKC",
                aspect_item.evidence.strip(),
            )

            # --------------------------------------------------
            # 3. Minimum evidence length
            # --------------------------------------------------
            MIN_EVIDENCE_LENGTH = 2  # Allows short, high-value Korean phrases like "찐맛집"

            if len(evidence_norm) < MIN_EVIDENCE_LENGTH:
                logger.warning("[Validator] Sub-length quote dropped ""(ID=%s): %r", raw_review.id, evidence_norm)
                continue

            # --------------------------------------------------
            # 4. Deterministic grounding
            # --------------------------------------------------
            if evidence_norm not in raw_text_norm:
                logger.warning("[Validator] Ungrounded quote dropped ""(ID=%s): %r", raw_review.id, evidence_norm)
                continue

            # --------------------------------------------------
            # 5. Deduplication
            # --------------------------------------------------
            dedup_key = (
                aspect_item.aspect,
                aspect_item.sentiment,
                evidence_norm,
            )

            if dedup_key in seen:
                logger.warning("[Validator] Duplicate aspect dropped ""(ID=%s): %r", raw_review.id, dedup_key)
                continue

            seen.add(dedup_key)
            validated_aspects.append(aspect_item)

        # ------------------------------------------------------
        # 6. Return new validated object
        # ------------------------------------------------------
        return ReviewABSA(
            review_id=raw_review.id,
            aspects=validated_aspects,
            summary=parsed_review.summary,
        )