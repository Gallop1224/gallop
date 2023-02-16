import numpy as np


def guidedfilter(src, I, r, eps):
    """
    guidedfilter: O(1) time implementation of guided filter.

    ----------
    :param src: filtering input image (should be a gray-scale/single channel image)
    :param I: guidance image (should be a gray-scale/single channel image)
    :param r: local window radius
    :param eps: regularization parameter
    :return: dst
    """
    (hei, wid) = I.shape[0:2]
    N = boxfilter(np.ones((hei, wid)), r)
    mean_I = np.divide(boxfilter(I, r), N)
    mean_p = np.divide(boxfilter(src, r), N)
    mean_Ip = np.divide(boxfilter(np.multiply(I, src), r), N)
    cov_Ip = mean_Ip - np.multiply(mean_I, mean_p)

    mean_II = np.divide(boxfilter(np.multiply(I, I), r), N)
    var_I = mean_II - np.multiply(mean_I, mean_I)

    a = np.divide(cov_Ip, (var_I + eps))
    b = mean_p - np.multiply(a, mean_I)

    mean_a = np.divide(boxfilter(a, r), N)
    mean_b = np.divide(boxfilter(b, r), N)

    dst = np.multiply(mean_a, I) + mean_b
    return dst


def boxfilter(src, r):
    """
    boxfilter: O(1) time box filtering using cumulative sum.

    ----------
    :param src: input (should be a gray-scale/single channel image)
    :param r: local window radius
    :return: dst(x, y)=sum(sum(src(x-r:x+r,y-r:y+r)))
    """
    (hei, wid) = src.shape[0:2]
    dst = np.zeros((hei, wid))

    cum = np.cumsum(src, axis=0)
    dst[0:r + 1, :] = cum[r:2 * r + 1, :]
    dst[r + 1:hei - r, :] = cum[2 * r + 1:hei, :] - cum[0:hei - 2 * r - 1, :]
    dst[hei - r:hei, :] = np.repmat(cum[hei - 1:hei, :], r, 1) - cum[hei - 2 * r - 1:hei - r - 1, :]

    cum = np.cumsum(dst, axis=1)
    dst[:, 0: r + 1] = cum[:, r: 2 * r + 1]
    dst[:, r + 1:wid - r] = cum[:, 2 * r + 1:wid] - cum[:, 0:wid - 2 * r - 1]
    dst[:, wid - r:wid] = np.repmat(cum[:, wid - 1:wid], 1, r) - cum[:, wid - 2 * r - 1:wid - r - 1]
    return dst
