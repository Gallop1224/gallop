from guidedfilter import *
import numpy as np

# 用了暗通道原方法去求t

def transmission(src, A, r, w, L):
    """

    :param src: original input image(three channels)
    :param A: airlight(1,1,3)
    :param r: radius of darkchannel
    :param w: t = 1 - w*(I/A)
    :return: dst
    """
    (hei, wid) = src.shape[0:2]
    tmp = np.zeros((hei, wid))
    for i in range(hei):
        for j in range(wid):
            tmp[i, j] = min(src[i, j, :] / A[0, 0, :])
    min_tmp = min_or_max(tmp, r, "min")
    dst = np.ones((hei, wid)) - min_tmp[:, :]

    dst  = np.vectorize(lambda x: x if x > 0.1 else 0.1)(dst)

    dst_refined = guidedfilter(dst, L, 30, 1e-6)
    return dst_refined


def min_or_max(src, r, op):
    """

    :param src: input(one channel only)
    :param r: window radius
    :param op: "min" or "max"
    :return: dst
    """
    if op == "min":
        operation = np.amin
    else:
        operation = np.amax
    (hei, wid) = src.shape[0:2]
    dst = np.zeros((hei, wid))
    src_pad = padding(src, r)
    for i in range(hei):
        for j in range(wid):
            dst[i, j] = operation(src_pad[i:i+2*r+1, j:j+2*r+1])
    return dst

def padding(src, r):
    """
    padding: used to padding the edges of the src with width of r

    --------
    :param src: input(one channel only)
    :param r: the width of padding
    :return: dst
    """
    (hei, wid) = src.shape[0:2]
    dst = np.zeros((hei+2*r, wid+2*r))
    dst[r:r+hei, r:r+wid] = src[:,:]
    dst[0:r, r:r + wid] = np.tile(src[0:1, 0:wid], [r,1])
    dst[r+hei:, r:r+wid] = np.tile(src[hei-1:hei, 0:wid], [r,1])
    dst[:, 0:r] = np.tile(dst[:, r:r+1], [1,r])
    dst[:, r+wid:] = np.tile(dst[:, r+wid-1:r+wid], [1,r])
    return dst