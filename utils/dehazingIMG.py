import numpy as np

def dehazingIMG(src, A, t):
    """
    J = (I(x) - A) / t(x) +A

    --------
    :param src: original input(three channels)
    :param A: airlight(1,1,3)
    :param trans: transmission(one channel only)
    :return: dst
    """
    (hei, wid) = src.shape[0:2]
    dst = np.zeros((hei, wid, 3))
    dst[:, :, 0] = np.divide((src[:, :, 0] - A[0, 0, 0]), t[:, :]) + A[0, 0, 0]
    dst[:, :, 1] = np.divide((src[:, :, 1] - A[0, 0, 1]), t[:, :]) + A[0, 0, 1]
    dst[:, :, 2] = np.divide((src[:, :, 2] - A[0, 0, 2]), t[:, :]) + A[0, 0, 2]
    dst = np.vectorize(lambda x: x if x < 1 else 1)(dst)
    dst = np.vectorize(lambda x: x if x > 0 else 0)(dst)
    return dst