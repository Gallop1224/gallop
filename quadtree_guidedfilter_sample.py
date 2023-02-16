import numpy as np
import scipy.ndimage
from utils.quadtree import airlight_quadtree
from utils.dehazingIMG import dehazingIMG
from utils.transmission_he import transmission

def defogging(src, img):
    L = np.array(img.convert("L")).astype(float) / 255
    src_d = scipy.ndimage.zoom(src, (0.5,0.5,1))
    L_d = scipy.ndimage.zoom(L, 0.5)
    (hei, wid) = src_d.shape[0:2]
    A = airlight_quadtree(src_d, L_d, 0.2)
    trans_d = transmission(src_d, A, round(0.02 * min(hei, wid)), 0.95, L_d)
    trans = np.reshape(scipy.ndimage.zoom(trans_d, 2), src.shape[0:2])
    dst = dehazingIMG(src, A, trans)
    return dst
