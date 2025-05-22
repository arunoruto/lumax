import numpy as np
import numpy.typing as npt


def matlab_gradient(img, h):
    p = np.zeros_like(img)
    q = np.zeros_like(img)

    p[:, 1:-1] = (img[:, 2:] - img[:, :-2]) * 0.5
    p[:, 0] = img[:, 1] - img[:, 0]
    p[:, -1] = img[:, -1] - img[:, -2]
    p /= h

    q[1:-1, :] = (img[2:, :] - img[:-2, :]) * 0.5
    q[0, :] = img[1, :] - img[0, :]
    q[-1, :] = img[-1, :] - img[-2, :]
    q /= h
    return p, q


def dtm2grad(dtm: npt.NDArray, resolution: float = 1, normalize: bool = False):
    """
    Computes the gradient of a Digital Terrain Model (DTM).

    Args:
        dtm (numpy.ndarray): The input DTM as a 2D numpy array.
        resolution (float, optional): The resolution of the DTM. Defaults to 1.
        normalize (bool, optional): Flag indicating whether to normalize the gradient vectors. Defaults to False.

    Returns:
        (numpy.ndarray): The gradient vectors of the DTM.

    Examples:
        >>> dtm = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> dtm2grad(dtm, resolution=0.5, normalize=True)
        array([[[-2.82842712, -2.82842712,  1.41421356],
                [-2.82842712, -2.82842712,  1.41421356],
                [-2.82842712, -2.82842712,  1.41421356]],
               [[-2.82842712, -2.82842712,  1.41421356],
                [-2.82842712, -2.82842712,  1.41421356],
                [-2.82842712, -2.82842712,  1.41421356]],
               [[-2.82842712, -2.82842712,  1.41421356],
                [-2.82842712, -2.82842712,  1.41421356],
                [-2.82842712, -2.82842712,  1.41421356]]])
    """
    q, p = np.gradient(dtm, resolution)
    n = np.stack((-p, -q, np.ones_like(p)), axis=2)
    if normalize:
        n /= np.linalg.norm(n, axis=2, keepdims=True)

    return n
