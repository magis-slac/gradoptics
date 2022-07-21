import torch


def batch_vector(x, y, z):
    """
    Returns a batch of vector

    :param x: Component of the vectors along the x axis (:obj:`torch.tensor`)
    :param y: Component of the vectors along the y axis (:obj:`torch.tensor`)
    :param z: Component of the vectors along the z axis (:obj:`torch.tensor`)

    :return: A batch of 3d vectors (:obj:`torch.tensor`)
    """
    return torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)


def normalize_batch_vector(input_vector, eps=1e-15):
    """
    Normalize the ``input_vector`` using the Euclidian norm

    :param input_vector: A batch of 3d vectors (:obj:`torch.tensor`)
    :param eps: Parameter used for numerical stability. Default is ``'1e-15'``
    """
    vector_norm = torch.sqrt(input_vector[:, 0] ** 2 + input_vector[:, 1] ** 2 + input_vector[:, 2] ** 2)
    x = (input_vector[:, 0] / (vector_norm + eps)).reshape(-1, 1)
    y = (input_vector[:, 1] / (vector_norm + eps)).reshape(-1, 1)
    z = (input_vector[:, 2] / (vector_norm + eps)).reshape(-1, 1)
    return torch.cat((x, y, z), dim=1)


def dot_product(batch_vector1, batch_vector2):
    """
    Computes the dot product between the given batch of 3d vectors

    :param batch_vector1: A batch of 3d vectors (:obj:`torch.tensor`)
    :param batch_vector2: A batch of 3d vectors (:obj:`torch.tensor`)

    :return: The dot product between ``batch_vector1`` and ``batch_vector2`` (:obj:`torch.tensor`)
    """
    return batch_vector1[:, 0] * batch_vector2[:, 0] + batch_vector1[:, 1] * batch_vector2[:, 1] + batch_vector1[:, 2] \
        * batch_vector2[:, 2]


def cos_theta(batch_vector1, batch_vector2, eps=1e-15):
    """
    Computes the cosine between the given batch of 3d vectors

    :param batch_vector1: A batch of 3d vectors (:obj:`torch.tensor`)
    :param batch_vector2: A batch of 3d vectors (:obj:`torch.tensor`)
    :param eps: Parameter used for numerical stability. Default is ``'1e-15'``

    :return: The cosine between ``batch_vector1`` and ``batch_vector2`` (:obj:`torch.tensor`)
    """
    norm_v1 = torch.sqrt(batch_vector1[:, 0] ** 2 + batch_vector1[:, 1] ** 2 + batch_vector1[:, 2] ** 2)
    norm_v2 = torch.sqrt(batch_vector2[:, 0] ** 2 + batch_vector2[:, 1] ** 2 + batch_vector2[:, 2] ** 2)
    return dot_product(batch_vector1, batch_vector2) / (norm_v1 * norm_v2 + eps)


def cross_product(vector1, vector2):
    """
    Computes the cross product between the given 3d vectors

    :param vector1: A 3d vector (:obj:`torch.tensor`)
    :param vector2: A 3d vector (:obj:`torch.tensor`)

    :return: The cross product between ``vector1`` and ``vector2`` (:obj:`torch.tensor`)
    """
    x = vector1[1] * vector2[2] - vector1[2] * vector2[1]
    y = vector1[2] * vector2[0] - vector1[0] * vector2[2]
    z = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    return torch.tensor([x, y, z])


def vector(x, y, z):
    """
    Returns a 3d vector

    :param x: Component of the vector along the x axis (:obj:`torch.tensor`)
    :param y: Component of the vector along the y axis (:obj:`torch.tensor`)
    :param z: Component of the vector along the z axis (:obj:`torch.tensor`)

    :return: A 3d vector (:obj:`torch.tensor`)
    """
    return torch.tensor([x, y, z], dtype=torch.float32)


def normalize_vector(input_vector, eps=1e-15):
    """
    Normalize the ``input_vector`` using the Euclidian norm

    :param input_vector: A 3d vector (:obj:`torch.tensor`)
    :param eps: Parameter used for numerical stability. Default is ``'1e-15'``
    """
    vector_norm = torch.sqrt(input_vector[0] ** 2 + input_vector[1] ** 2 + input_vector[2] ** 2)
    return input_vector / (vector_norm + eps)


def get_perpendicular_vector(input_vector, eps=1e-15):
    """
    Returns a vector perpendicular to ``input_vector``

    :param input_vector: A 3d vector (:obj:`torch.tensor`)
    :param eps: Parameter used for numerical stability. Default is ``'1e-15'``
    """
    x1 = 1.
    y1 = 1.
    # x1 * input_vector[0] + y1 * input_vector[1] + z1 * input_vector[2] = 0
    z1 = - (x1 * input_vector[0] + y1 * input_vector[1]) / (input_vector[2] + eps)
    return vector(x1, y1, z1)
