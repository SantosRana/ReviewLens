# src/absa.py
import logging
from typing import List, Dict, Any
from src.groq_client import GroqEngine
from src.schema import ReviewABSA, ReviewInput
from src.validator import ABSAOutputValidator

logger = logging.getLogger(__name__)


class ABSAOrchestrator:
    def __init__(self, batch_size: int = 5):
        self.engine = GroqEngine()
        self.batch_size = batch_size
        self.validator = ABSAOutputValidator()

    def _chunk_list(self, items: List[Any], size: int):
        """Yield fixed-size batches while preserving ReviewInput IDs."""
        for i in range(0, len(items), size):
            yield items[i:i + size]

    def process_reviews(self, raw_reviews: List[str]) -> Dict[str, Any]:
        if not raw_reviews:
            return self._empty_result()

        # Canonical review objects
        review_inputs = [
            ReviewInput(id=i + 1, text=text)
            for i, text in enumerate(
                txt for txt in raw_reviews
                if txt and txt.strip()
            )
        ]

        if not review_inputs:
            return self._empty_result()

        input_map = {
            review.id: review
            for review in review_inputs
        }

        validated_details: List[ReviewABSA] = []

        stats = {
            "retrieved": len(raw_reviews),
            "analyzed": len(review_inputs),
            "raw_aspects": 0,
            "validated_aspects": 0,
            "rejected_aspects": 0,
            "unknown_ids": 0,
        }

        for batch in self._chunk_list(review_inputs, self.batch_size):
            try:
                batch_response = self.engine.analyze_reviews_batch(batch)

                for rev in batch_response.reviews:
                    stats["raw_aspects"] += len(rev.aspects)

                    target_review = input_map.get(rev.review_id)

                    if target_review is None:
                        logger.error(
                            "[ABSA] Unknown review_id=%s returned by LLM",
                            rev.review_id,
                        )
                        stats["unknown_ids"] += 1
                        continue

                    validated_rev = self.validator.validate_review(
                        parsed_review=rev,
                        raw_review=target_review,
                    )

                    stats["validated_aspects"] += len(
                        validated_rev.aspects
                    )

                    stats["rejected_aspects"] += (
                        len(rev.aspects)
                        - len(validated_rev.aspects)
                    )

                    validated_details.append(validated_rev)

            except Exception:
                logger.exception("[ABSA] Batch processing error")

        validation_retention_rate = (
            stats["validated_aspects"]
            / stats["raw_aspects"]
            * 100
            if stats["raw_aspects"] > 0
            else 0.0
        )

        logger.info(
            "[ABSA Metrics] Retrieved=%d | Analyzed=%d | "
            "Raw=%d | Validated=%d | Rejected=%d | "
            "Retention=%.1f%% | UnknownIDs=%d",
            stats["retrieved"],
            stats["analyzed"],
            stats["raw_aspects"],
            stats["validated_aspects"],
            stats["rejected_aspects"],
            validation_retention_rate,
            stats["unknown_ids"],
        )

        return self._aggregate_results(
            validated_details=validated_details,
            total_reviews=stats["retrieved"],
            metrics=stats,
            grounding_rate=validation_retention_rate,
        )

    def _aggregate_results(
        self,
        validated_details: List[ReviewABSA],
        total_reviews: int,
        metrics: Dict[str, Any],
        grounding_rate: float,
    ) -> Dict[str, Any]:
        """Aggregates aspect metrics, overall score, pros/cons, and pipeline metrics."""
        metrics_summary = self._aggregate_metrics(validated_details)

        return {
            "status": "success",
            "total_reviews_analyzed": total_reviews,
            "overall_score": metrics_summary["overall_score"],
            "aspect_scores": metrics_summary["aspects"],
            "pros": metrics_summary["pros"],
            "cons": metrics_summary["cons"],
            "details": validated_details,
            "grounding_rate": round(grounding_rate, 1),
            "pipeline_metrics": {
                **metrics,
                "grounding_rate_pct": round(grounding_rate, 1),
            },
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Fallback payload when 0 reviews are fetched or analyzed."""
        return {
            "status": "empty",
            "total_reviews_analyzed": 0,
            "overall_score": 0.0,
            "aspect_scores": {},
            "pros": [],
            "cons": [],
            "details": [],
            "pipeline_metrics": {
                "retrieved": 0,
                "analyzed": 0,
                "raw_aspects": 0,
                "validated_aspects": 0,
                "rejected_aspects": 0,
                "unknown_ids": 0,
                "grounding_rate_pct": 0.0,
            },
        }

    def _aggregate_metrics(self, parsed_reviews: List[ReviewABSA]) -> Dict[str, Any]:
        aspect_counts = {
            "FOOD": {"positive": 0, "negative": 0, "neutral": 0},
            "PRICE": {"positive": 0, "negative": 0, "neutral": 0},
            "SERVICE": {"positive": 0, "negative": 0, "neutral": 0},
            "AMBIENCE": {"positive": 0, "negative": 0, "neutral": 0},
        }

        pros, cons = [], []

        for rev in parsed_reviews:
            for item in rev.aspects:
                if item.aspect in aspect_counts:
                    aspect_counts[item.aspect][item.sentiment] += 1

                    if item.sentiment == "positive" and item.evidence not in pros:
                        pros.append(item.evidence)
                    elif item.sentiment == "negative" and item.evidence not in cons:
                        cons.append(item.evidence)

        aspect_scores = {}
        total_positive = 0
        total_neutral = 0
        total_mentions = 0

        for aspect, counts in aspect_counts.items():
            total = counts["positive"] + counts["negative"] + counts["neutral"]
            if total > 0:
                pos_ratio = round((counts["positive"] / total) * 100, 1)
                aspect_scores[aspect] = {
                    "score": pos_ratio,
                    "positive": counts["positive"],
                    "negative": counts["negative"],
                    "neutral": counts["neutral"],
                    "total_mentions": total,
                }
                total_positive += counts["positive"]
                total_neutral += counts["neutral"]
                total_mentions += total

        if total_mentions > 0:
            normalized_sentiment = (total_positive + (0.5 * total_neutral)) / total_mentions
            overall_rating = round(normalized_sentiment * 5.0, 2)
        else:
            overall_rating = 0.0

        return {
            "aspects": aspect_scores,
            "overall_score": overall_rating,
            "pros": pros[:5],
            "cons": cons[:5],
        }