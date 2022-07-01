import numpy as np


def ambient_lighting(light, light_intensity, light_color):
    if isinstance(light_color, tuple) or isinstance(light_color, list):
        light_color = np.array(light_color, dtype='float32')
    elif isinstance(light_color, np.ndarray):
        light_color = np.array(light_color).float()
    if len(light_color.shape) == 1:
        light_color = light_color[None, :]

    light += light_intensity * light_color[:, None, :]  # light_color:[1, 1, 3] -> light:[bs, nf, 3]
    return light