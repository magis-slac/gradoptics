import torch


def batch_vector(x, y, z):
    return torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)


def normalize_batch_vector(input_vector, eps=1e-15):
    vector_norm = torch.sqrt(input_vector[:, 0] ** 2 + input_vector[:, 1] ** 2 + input_vector[:, 2] ** 2)
    x = (input_vector[:, 0] / (vector_norm + eps)).reshape(-1, 1)
    y = (input_vector[:, 1] / (vector_norm + eps)).reshape(-1, 1)
    z = (input_vector[:, 2] / (vector_norm + eps)).reshape(-1, 1)
    return torch.cat((x, y, z), dim=1)


def dot_product(batch_vector1, batch_vector2):
    return batch_vector1[:, 0] * batch_vector2[:, 0] + \
           batch_vector1[:, 1] * batch_vector2[:, 1] + \
           batch_vector1[:, 2] * batch_vector2[:, 2]


def vector(x, y, z):
    return torch.tensor([x, y, z], dtype=torch.float32)


def normalize_vector(input_vector, eps=1e-15):
    vector_norm = torch.sqrt(input_vector[0] ** 2 + input_vector[1] ** 2 + input_vector[2] ** 2)
    return input_vector / (vector_norm + eps)


def get_perpendicular_vector(input_vector, eps=1e-15):
    """
    @Todo: docs (returned vector not normalized)
    :param input_vector:
    :param eps:
    :return:
    """
    x1 = 1.
    y1 = 1.
    # x1 * input_vector[0] + y1 * input_vector[1] + z1 * input_vector[2] = 0
    z1 = - (x1 * input_vector[0] + y1 * input_vector[1]) / (input_vector[2] + eps)
    return vector(x1, y1, z1)
