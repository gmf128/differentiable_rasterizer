import numpy as np
from numba import cuda

@cuda.jit()
def texture_map_cuda(image, texture_res, textures, faces, is_update, width, height, texture_size):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if (i * 3 >= texture_size) :
        return
    R = texture_res
    fn = int(i / (R * R))
    w_y = int((i % (R * R)) / R)
    w_x = int(i % R)
    # compute  barycoordinate w0, w1, w2;
    if (w_x + w_y < R) :
        w0 = (w_x + 1. / 3.) / R
        w1 = (w_y + 1. / 3.) / R
        w2 = 1. - w0 - w1
    else :
        w0 = ((R - 1. - w_x) + 2. / 3.) / R
        w1 = ((R - 1. - w_y) + 2. / 3.) / R
        w2 = 1. - w0 - w1

    if (is_update[fn] == 0):
        return


    pos_x = ((faces[fn, 0, 0] * w0 + faces[fn, 1, 0] * w1 + faces[fn, 2, 0] * w2) * (width - 1))
    pos_y = ((faces[fn, 0, 1] * w0 + faces[fn, 1, 1] * w1 + faces[fn, 2, 1] * w2) * (height - 1))

    weight_x1 = pos_x - int(pos_x)
    weight_x0 = 1 - weight_x1
    weight_y1 = pos_y - int(pos_y)
    weight_y0 = 1 - weight_y1
    for k in range(0, 3):
        c = 0
        c += image[int(pos_x), int(pos_y), k] * (weight_x0 * weight_y0)
        c += image[int(pos_x), int(pos_y)+1, k] * (weight_x0 * weight_y1)
        c += image[int(pos_x)+1, int(pos_y), k] * (weight_x1 * weight_y0)
        c += image[int(pos_x)+1, int(pos_y)+1, k] * (weight_x1 * weight_y1)
        textures[fn, w_y*R+w_x, k] = c / 255.





def texture_mapping(texture_img, texture_res, textures, faces, is_update, width, height):
    assert len(textures.shape) == 3
    texture_size = textures.shape[0] * textures.shape[1] * textures.shape[2]  # nf * res^2 * 3
    ThreadsperBlock = 1024
    BlocksperGrid = int(texture_size/3/ThreadsperBlock)+1
    texture_img = cuda.to_device(np.ascontiguousarray(texture_img))
    textures = cuda.to_device(textures)
    faces = cuda.to_device(faces)
    is_update = cuda.to_device(is_update)
    texture_map_cuda[BlocksperGrid, ThreadsperBlock](texture_img, texture_res, textures, faces, is_update, width, height, texture_size)
    cuda.synchronize()
    textures = textures.copy_to_host()
    return textures