import math
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

img_dir = os.listdir('../out/out_My')
print(img_dir)
for oneimg in img_dir:

    path = '../out/out_My/' + oneimg
    # gray= cv2.imread('../img/3.jpg', cv2.IMREAD_GRAYSCALE)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    [rows, cols] = gray.shape
    # RGB 3 channels
    # hist1 = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
    # hist2 = cv2.calcHist([img],[1],None,[256],[0.0,255.0])
    # hist3 = cv2.calcHist([img],[2],None,[256],[0.0,255.0])
    # 直方圖
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0.0, 255.0])
    plt.plot(range(256), hist_gray)

    # hn valueis not correct
    hb = np.zeros((256, 1), np.float32)
    # hn = np.zeros((256, 1), np.float32)
    for j in range(0, 256):
        hb[j, 0] = hist_gray[j, 0] / (rows * cols)
    # plt.plot(range(256), hb)
    '''
    c = np.zeros((256, 1), np.float32)
    c[0, 0] = hb[0, 0]
    for l in range(1, 256):
        c[l, 0] = c[l-1, 0] + hb[l, 0]
    '''
    H = 0
    for i in range(0, 256):
        if hb[i, 0] > 0:
            H = H - (hb[i, 0]) * math.log2(hb[i, 0])

    entropy = H

    print(oneimg,entropy)