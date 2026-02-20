import numpy as np


def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)

    if max_len is None:
        L = max([len(seq) for seq in seqs])
    else:
        L = max_len

    # creating an empty matrix of size N * L with give pad_value
    padding = np.full((N, L), pad_value)
    # Iterating each row trunc it with max length adding it to first (right padding)
    for i, seq in enumerate(seqs):
        trunc = seq[:L]
        padding[i, : len(trunc)] = trunc
    return padding
