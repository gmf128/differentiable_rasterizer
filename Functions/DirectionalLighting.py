import numpy as np
import jittor as jt

def directional_lighting(light, normals, light_intensity=0.5, light_color=(1, 1, 1),
                         light_direction=(0, 1, 0)):
    # normals: [nb, :, 3]

    if isinstance(light_color, tuple) or isinstance(light_color, list):
        light_color = np.array(light_color, dtype='float32')
    elif isinstance(light_color, np.ndarray):
        light_color = np.array(light_color).float()
    if isinstance(light_direction, tuple) or isinstance(light_direction, list):
        light_direction = np.array(light_direction, dtype='float32')
    elif isinstance(light_direction, np.ndarray):
        light_direction = np.array(light_direction).float()
    if len(light_color.shape) == 1:
        light_color = light_color[None, :]
    if len(light_direction.shape) == 1:
        light_direction = light_direction[None, :]  # [1, 3]

    cosine = jt.nn.relu(jt.sum(normals * light_direction, dim=2))  # []
    light += light_intensity * (light_color[:, None, :] * cosine[:, :, None])
    return light  # [nb, :, 3]