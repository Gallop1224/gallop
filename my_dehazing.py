import copy
import math
import os
import random
import time

import cv2
import numpy as np
import scipy

'''
四叉树找Ａ
'''


def AirlightEstimate(img, windowsize=15):
    # start = time.time()
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pix = min(im.shape)
    n = math.log2(pix/windowsize)
    # print(n)
    n = int(n)-1
    # print(n)
    imlist = [[], []]
    for times in range(3):
        h, w = im.shape[0], im.shape[1]
        imlist[0].append(im[0:int(w / 2), 0:int(h / 2)])
        imlist[0].append(im[int(w / 2):w, 0:int(h / 2)])
        imlist[0].append(im[0:int(h / 2), int(h / 2):h])
        imlist[0].append(im[int(w / 2):w, int(h / 2):h])
        imlist[1].append(img[0:int(w / 2), 0:int(h / 2)])
        imlist[1].append(img[int(w / 2):w, 0:int(h / 2)])
        imlist[1].append(img[0:int(h / 2), int(h / 2):h])
        imlist[1].append(img[int(w / 2):w, int(h / 2):h])
        scorelist = [np.mean(i) for i in imlist[0]]
        max_index = np.argmax(scorelist)
        im, img = imlist[0][max_index], imlist[1][max_index]
        imlist = [[], []]
    img_single_channel_list = cv2.split(img)
    A = []
    for channel in img_single_channel_list:
        A.append(np.mean(channel))

    return A
    # end = time.time()
    # print(A)
    # print("AirlightEstimate运行时间：%.3f秒" % (end - start))
    # print(1/(end - start))


'''
基于边界限制算粗透射率 windowSze = 3
                  C0 = 20  # Default value = 20 (as recommended in the paper)
                  C1 = 300  # Default value = 300 (as recommended in the paper)
'''


def BoundaryConstraints(HazeImg, A, C0=20, C1=300, windowSze=3):
    if len(HazeImg.shape) == 3:
        t_b = np.maximum((A[0] - HazeImg[:, :, 0].astype(np.float64)) / (A[0] - C0),
                         # np.maximum(X, Y) 用于逐元素比较两个array的大小。
                         (HazeImg[:, :, 0].astype(np.float64) - A[0]) / (C1 - A[0]))  # max( (A-I(X))/(A-C0),
        t_g = np.maximum((A[1] - HazeImg[:, :, 1].astype(np.float64)) / (A[1] - C0),  # (A-I(X))/(A-C1) )
                         (HazeImg[:, :, 1].astype(np.float64) - A[1]) / (C1 - A[1]))
        t_r = np.maximum((A[2] - HazeImg[:, :, 2].astype(np.float64)) / (A[2] - C0),
                         (HazeImg[:, :, 2].astype(np.float64) - A[2]) / (C1 - A[2]))

        MaxVal = np.maximum(t_b, t_g, t_r)
        transmission = np.minimum(MaxVal, 1)  # min{ max( (A-I(X))/(A-C0),(A-I(X))/(A-C1) ) , 1 }
    else:  # 单通道
        transmission = np.maximum((A[0] - HazeImg.astype(np.float64)) / (A[0] - C0),
                                  (HazeImg.astype(np.float64) - A[0]) / (C1 - A[0]))
        transmission = np.minimum(transmission, 1)

    kernel = np.ones((windowSze, windowSze), np.float64)
    tbx = cv2.morphologyEx(transmission, cv2.MORPH_CLOSE, kernel=kernel)  # 先进行膨胀操作，再进行腐蚀操作
    return tbx  # 可以填充小的黑洞（fill hole补洞），去掉小的黑噪点。填充目标区域内的离散小空洞和分散部分


'''
导向滤波细化透射率
 r = 60;
 eps = 0.0001;
 30, 0.1
 r=40, eps=1e-3
'''


def guidedfilter(src_gray, I, r, eps):
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
    mean_p = np.divide(boxfilter(src_gray, r), N)
    mean_Ip = np.divide(boxfilter(np.multiply(I, src_gray), r), N)
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


def Guidedfilter_cv(im_gray, p, r, eps):
    start = time.time()
    mean_I = cv2.boxFilter(im_gray, cv2.CV_64F, (r, r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
    mean_Ip = cv2.boxFilter(im_gray * p, cv2.CV_64F, (r, r));
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im_gray * im_gray, cv2.CV_64F, (r, r));
    var_I = mean_II - mean_I * mean_I;

    a = cov_Ip / (var_I + eps);
    b = mean_p - a * mean_I;

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

    q = mean_a * im_gray + mean_b;
    end = time.time()
    # print("Guidedfilter运行时间：%.3f秒" % (end - start))
    return q;


def recover(imgar, A, t, tmin=0.1):
    """
    Radiance recovery. According to section (4.3) and equation (16) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imgar:    an H*W RGB hazed image
    atm:      the atmospheric light in imgar
    t:        the transmission in imgar
    tmin:     the minimum value that transmission can take (default=0.1)

    Return
    -----------
    The imaged recovered and dehazed, j (a H*W RGB matrix).
    """

    # the output dehazed image
    j = np.zeros(imgar.shape)

    # equation (16)
    for c in range(0, imgar.shape[2]):
        j[:, :, c] = ((imgar[:, :, c].astype(float) - A[c]) / np.maximum(t[:, :], tmin)) + A[c]
        # j = np.maximum(np.minimum(j, 255), 0)

    return j / np.amax(j)


def recover_meng(img, A, t):
    HazeCorrectedImage = copy.deepcopy(img)
    epsilon = 0.0001
    delta = 0.6
    t = pow(np.maximum(abs(t), epsilon), delta)
    for ch in range(len(img.shape)):
        temp = ((img[:, :, ch].astype(float) - A[ch]) / t) + A[ch]
        temp = np.maximum(np.minimum(temp, 255), 0)
        HazeCorrectedImage[:, :, ch] = temp
    return HazeCorrectedImage


def show(img):
    cv2.imshow('result', img)
    cv2.waitKey(0)  # 按任意键继续执行，可以自定义设置时间，单位毫秒
    # cv2.destroyAllWindows()


def LUT(img, brightness):
    factor = 1.0 + random.uniform(-1.0 * brightness, brightness)
    table = np.array([(i / 255.0) * factor * 255 for i in np.arange(0, 256)]).clip(0, 255).astype(np.uint8)
    image = cv2.LUT(img, table)
    i = cv2.L
    return image


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def img_contrast_bright(img):
    a = 1.3
    b = 1 - a
    g = 0.1
    h, w, c = img.shape
    blank = np.zeros([h, w, c], img.dtype)
    dst = cv2.addWeighted(img, a, blank, b, g)
    return dst


img_dir = os.listdir('./imgHazeKS')
print(img_dir)
for oneimg in img_dir:
    path = './imgHazeKS/' + oneimg
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    A = AirlightEstimate(img)
    # 这里给A一个限制
    # limit = 0.9
    # b=[limit,limit,limit]
    # A = np.multiply(A,b)
    # print(A)
    tb = BoundaryConstraints(img, A)
    consttime = time.time()
    # t = guidedfilter(img_gray, tb, 60, 0.0001)
    t = Guidedfilter_cv(img_gray, tb, 30, 0.0001)

    # out = Recover(img, t, A)

    # out2 = recover(img,A,t )
    # show(out2)
    # out2 = cv2.normalize(out2,None,0,255,cv2.NORM_MINMAX)
    out_meng = recover_meng(img, A, t)
    end = time.time()

    cv2.imwrite('./outKS/out_My/' + oneimg, out_meng)
    # show(out_meng)

    # cv2.imwrite('out/out1_meng.jpg', out_meng)
    # out3 = img_contrast_bright(out2)
    # # meng
    # # HazeCorrectedImage = copy.deepcopy(img)
    # # for ch in range(len(img.shape)):
    # #     temp = ((img[:, :, ch].astype(float) - A[ch]) / t) + A[ch]
    # #     temp = np.maximum(np.minimum(temp, 255), 0)
    # #     HazeCorrectedImage[:, :, ch] = temp

    # print(path, '的时间', (consttime - start))
    #print(oneimg, (consttime - start))
    print( (consttime - start))
    # print("算A与tb %.3f" % (consttime - start))
    # print(end - start)
    # print(1 / (end - start))
