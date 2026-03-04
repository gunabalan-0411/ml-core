import numpy as np


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    np_matrix = np.array(matrix)
    axis = 0 if mode == "column" else 1
    return np_matrix.mean(axis=axis).tolist()
