import time
from PIL import Image
import cv2
import numpy as np
from scipy import ndimage


def airlight_quadtree(src,  ratio=0.2):
    """
    airlight: used to calculate values of airlight

    --------
    :param src: original input image with 3 channels
    :param L: lumination of the src, with only 1 channel
    :param ratio: the final window radius < ratio * min(height, width)
    :return: al(1,1,3)
    """

    im = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    L = np.array(im).astype(float) / 255  # convert转为灰度图，转为float，创建np数组，再除以255
    src_d = ndimage.zoom(src, (0.5,0.5,1))        # 对图像插值
    L_d = ndimage.zoom(L, 0.5)
    (hei, wid) = src_d.shape[0:2]

    # (hei, wid) = src.shape[0:2]
    min_r = ratio * min((hei, wid))
    al = quadtree(src_d, L_d, min_r)
    return al
# 四叉树找A
def quadtree(src, L, min_r):
    (hei, wid) = src.shape[0:2]
    if hei < min_r or wid < min_r:
        al = np.zeros((1,1,3))
        al[0, 0, 0] = np.mean(src[:, :, 0])
        al[0, 0, 1] = np.mean(src[:, :, 1])
        al[0, 0, 2] = np.mean(src[:, :, 2])
        return al
    mid_hei = np.floor(hei / 2).astype(int)
    mid_wid = np.floor(wid / 2).astype(int)
    quad1 = np.mean(L[0:mid_hei, 0:mid_wid])
    quad2 = np.mean(L[mid_hei:hei, 0:mid_wid])
    quad3 = np.mean(L[0:mid_hei, mid_wid:wid])
    quad4 = np.mean(L[mid_hei:hei, mid_wid:wid])
    if quad1 >= max(quad2, quad3, quad4):
        al = quadtree(src[0:mid_hei, 0:mid_wid, :], L[0:mid_hei, 0:mid_wid], min_r)
    elif quad2 >= max(quad3, quad4):
        al = quadtree(src[mid_hei:hei, 0:mid_wid, :], L[mid_hei:hei, 0:mid_wid], min_r)
    elif quad3 >= quad4:
        al = quadtree(src[0:mid_hei, mid_wid:wid, :], L[0:mid_hei, mid_wid:wid], min_r)
    else:
        al = quadtree(src[mid_hei:hei, mid_wid:wid, :], L[mid_hei:hei, mid_wid:wid], min_r)
    return al


path = '../img/1.jpg'
img =cv2.imread(path)
start = time.time()
A  = airlight_quadtree(img)
end = time.time()

print(A)
print(end-start)
print(1/(end-start))