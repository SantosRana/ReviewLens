# kc_electra/model.py
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from .decode_prediction import decode_predictions


class KcElectraSharedSentiment(nn.Module):
    """Shared-encoder model for multi-aspect sentiment analysis."""

    def __init__(self, model_name: str, num_aspects: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.mention_head = nn.Linear(hidden, num_aspects)
        self.sentiment_heads = nn.ModuleList(
            [nn.Linear(hidden, 2) for _ in range(num_aspects)]
        )

    def forward(self, input_ids=None, attention_mask=None) -> dict:
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = self.dropout(encoder_out.last_hidden_state[:, 0])

        mention_logits = self.mention_head(cls)
        sentiment_logits = torch.stack(
            [head(cls) for head in self.sentiment_heads], dim=1
        )

        B = cls.size(0)
        packed = torch.cat(
            [mention_logits, sentiment_logits.reshape(B, -1)], dim=1
        )
        return {"logits": packed}


class SharedABSAWrapper:
    """Inference-only wrapper for the trained ABSA model."""

    def __init__(self, model_name: str, aspects: list, max_length: int):
        self.aspects = aspects
        self.max_length = max_length
        self.thresholds = {a: 0.5 for a in aspects}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = KcElectraSharedSentiment(
            model_name=model_name,
            num_aspects=len(aspects)
        )

    def predict(self, X) -> pd.DataFrame:
        """
        Run inference on input texts.

        Parameters
        ----------
        X : str or list of str

        Returns
        -------
        pd.DataFrame with columns = aspects
        Labels: 0 = Not Mentioned, 1 = Negative, 2 = Positive
        """
        self.model.eval()

        texts = [X] if isinstance(X, str) else list(X)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc.pop("token_type_ids", None)

        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)

        packed_logits = outputs["logits"].cpu().numpy()

        return decode_predictions(
            packed_logits,
            self.aspects,
            self.thresholds
        )