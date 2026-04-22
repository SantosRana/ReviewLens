# config.py
from pathlib import Path

NAVER_CLIENT_ID = "h5Wu7OzTZNq21VNfzaaF"
NAVER_CLIENT_SECRET = "9OTDKFYw0c"

ROOT = Path(__file__).parent
WEIGHTS_DIR = ROOT / "weights" / "kc_electra"

OLLAMA_MODEL = "qwen2.5:7b-instruct"

ASPECTS = ["FOOD", "PRICE", "SERVICE", "AMBIENCE"]
LABEL_MAP = {0: "not_mentioned", 1: "negative", 2: "positive"}