import numpy as np


def reshape_matrix(
    a: list[list[int | float]], new_shape: tuple[int, int]
) -> list[list[int | float]]:
    # Write your code here and return a python list after reshaping by using numpy's tolist() method
    a = np.array(a)
    if a.size != new_shape[0] * new_shape[1]:
        return []
    return a.reshape(new_shape).tolist()
