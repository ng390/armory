import numpy as np


def add_large_pattern_bd(
    x: np.ndarray, distance: int = 20, square_size: int = 8, pixel_value: int = 1
) -> np.ndarray:
    x = np.array(x)

    shape = x.shape
    if len(shape) == 4:
        width, height = x.shape[1:3]
        x[:, width - distance, height - distance, :] = pixel_value
        for i in range(square_size):
            for j in range(square_size):
                x[:, width - distance - i, height - distance - j, :] = pixel_value
    elif len(shape) == 3:
        width, height = x.shape[1:]
        for i in range(square_size):
            for j in range(square_size):
                x[:, width - distance - i, height - distance - j] = pixel_value
    else:
        raise ValueError("Invalid array shape: " + str(shape))
    return x
