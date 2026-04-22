import numpy as np

def _unpack_logits(packed: np.ndarray, num_aspects: int) -> tuple:
    """
    Split the packed model output into mention and sentiment logit arrays.

    Parameters
    ----------
    packed      : (N, 3A) numpy array from Trainer.predict
    num_aspects : int, number of aspects A

    Returns
    -------
    mention_logits   : (N, A)    pre-sigmoid scores for mention detection
    sentiment_logits : (N, A, 2) pre-softmax scores for sentiment
    """
    packed           = np.asarray(packed)
    mention_logits   = packed[:, :num_aspects]                              # (N, A)
    sentiment_logits = packed[:, num_aspects:].reshape(-1, num_aspects, 2) # (N, A, 2)
    return mention_logits, sentiment_logits


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid (numerically stable)."""
    return 1.0 / (1.0 + np.exp(-x))


