# absa.py
import torch
import json
from transformers import AutoTokenizer
from config import ROOT, WEIGHTS_DIR, ASPECTS, LABEL_MAP

from .kc_electra.model import SharedABSAWrapper
def load_model():
    """
    Load and initialize the sentiment analysis model.

    Returns:
        SharedABSAWrapper: Configured model ready for evaluation.
    """

    # Initialize model wrapper with configuration
    model = SharedABSAWrapper(
        model_name="beomi/KcELECTRA-base-v2022",
        aspects=ASPECTS,
        max_length=128,
    )

    # Load model weights from file
    state_dict = torch.load(WEIGHTS_DIR / "kc_electra.pt", map_location="cpu")
    state_dict.pop("mention_pos_weight", None)  # Remove unused weights
    state_dict.pop("sentiment_class_weights", None)
    model.model.load_state_dict(state_dict, strict=False)

    # Move model to CPU
    model.model.to("cpu")

    # Load tokenizer from directory
    model.tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

    # Load thresholds for sentiment classification
    with open(WEIGHTS_DIR / "thresholds.json", "r", encoding="utf-8") as f:
        model.thresholds = json.load(f)

    # Set model to evaluation mode
    model.model.eval()
    return model

import torch
import json
from transformers import AutoTokenizer
from config import ROOT, WEIGHTS_DIR, ASPECTS, LABEL_MAP

from .kc_electra.model import SharedABSAWrapper


def load_model():
    """
    Load and initialize the sentiment analysis model.

    Returns:
        SharedABSAWrapper: Configured model ready for evaluation.
    """

    # Initialize model wrapper with configuration
    model = SharedABSAWrapper(
        model_name="beomi/KcELECTRA-base-v2022",
        aspects=ASPECTS,
        max_length=128,
    )

    # Load model weights from file
    state_dict = torch.load(WEIGHTS_DIR / "kc_electra.pt", map_location="cpu")
    state_dict.pop("mention_pos_weight", None)  # Remove unused weights
    state_dict.pop("sentiment_class_weights", None)
    model.model.load_state_dict(state_dict, strict=False)

    # Move model to CPU
    model.model.to("cpu")

    # Load tokenizer from directory
    model.tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

    # Load thresholds for sentiment classification
    with open(WEIGHTS_DIR / "thresholds.json", "r", encoding="utf-8") as f:
        model.thresholds = json.load(f)

    # Set model to evaluation mode
    model.model.eval()
    return model


def analyze_reviews(kc_model, reviews, progress_callback=None) -> dict:
    """
    Analyze sentiment of reviews by aspect.

    Args:
        kc_model: Sentiment analysis model.
        reviews (list or str): Reviews to analyze.
        progress_callback: Optional callback(current, total, label) for progress updates.

    Returns:
        dict: Summary of sentiment counts and dominant sentiment per aspect.
    """

    # Normalise to list
    if isinstance(reviews, str):
        reviews = [reviews]

    counts = {
        aspect: {"positive": 0, "negative": 0, "not_mentioned": 0}
        for aspect in ASPECTS
    }

    total = len(reviews)
    for i, review in enumerate(reviews):
        # Predict one review at a time so we can report progress
        result_df = kc_model.predict([review])

        for _, row in result_df.iterrows():
            for aspect in counts:
                label = LABEL_MAP.get(int(row[aspect]), "not_mentioned")
                counts[aspect][label] += 1

        # Call progress callback with label parameter
        if progress_callback:
            progress_callback(i + 1, total, f"Processing review {i+1} of {total}...")

    # Build summary with dominant sentiment per aspect
    summary = {}
    for aspect, sentiments in counts.items():
        relevant = {
            k: v for k, v in sentiments.items()
            if k != "not_mentioned" and v > 0
        }
        if relevant:
            dominant = max(relevant, key=relevant.get)
            summary[aspect] = {
                "overall": dominant,
                "positive": sentiments["positive"],
                "negative": sentiments["negative"],
                "not_mentioned": sentiments["not_mentioned"]
            }

    return summary