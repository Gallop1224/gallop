import math
import time
import os
import random
import copy
import cv2
import numpy as np

def Airlight_He(HazeImg, AirlightMethod='fast', windowSize=15):
    if AirlightMethod.lower() == 'fast':
        A = []
        if len(HazeImg.shape) == 3:
            for ch in range(len(HazeImg.shape)):
                kernel = np.ones((windowSize, windowSize), np.uint8)  # [[1,1,1],[1,1,1],[1,1,1]] kernel
                minImg = cv2.erode(HazeImg[:, :, ch], kernel)  # 腐蚀操作 局部最小值
                A.append(int(minImg.max()))  # 首先使用最小滤波对输入图像每个通道进行滤波，然后取每个通道的最大值最为A分量的估计值
        else:  # 单通道图
            kernel = np.ones((windowSize, windowSize), np.uint8)
            minImg = cv2.erode(HazeImg, kernel)
            A.append(int(minImg.max()))
    return (A)
def AirlightEstimate(img, windowsize=50):
    # start = time.time()
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pix = min(im.shape)
    n = math.log2(pix/windowsize)
    # print(n)
    n = int(n)
    print(n)
    imlist = [[], []]
    for times in range(n):
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
    print(A)
    return A
    # end = time.time()
    # print(A)
    # print("AirlightEstimate运行时间：%.3f秒" % (end - start))
    # print(1/(end - start))

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

def recover_meng(img, A, t):
    HazeCorrectedImage = copy.deepcopy(img)
    epsilon = 0.0001
    delta = 0.7
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

def img_contrast_bright(img):
    a = 1.3
    b = 1 - a
    g = 30
    h, w, c = img.shape
    blank = np.zeros([h, w, c], img.dtype)
    dst = cv2.addWeighted(img, a, blank, b, g)
    return dst






path = './imgHazeKS/4.png'
img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
start = time.time()
# A = AirlightEstimate(img)
A = Airlight_He(img)
    # 这里给A一个限制
    # limit = 0.9
    # b=[limit,limit,limit]
    # A = np.multiply(A,b)
    # print(A)
tb = BoundaryConstraints(img, A)
consttime = time.time()
    # t = guidedfilter(img_gray, tb, 60, 0.0001)
t = Guidedfilter_cv(img_gray, tb, 50, 0.0001)

    # out = Recover(img, t, A)

    # out2 = recover(img,A,t )
    # show(out2)
    # out2 = cv2.normalize(out2,None,0,255,cv2.NORM_MINMAX)
out_meng = recover_meng(img, A, t)
end = time.time()


# show(out_meng)

    # cv2.imwrite('out/out1_meng.jpg', out_meng)
out3 = img_contrast_bright(out_meng)
show(out3)
#cv2.imwrite('./out/out_My/signal1.jpg' , out3)
# # meng
    # # HazeCorrectedImage = copy.deepcopy(img)
    # # for ch in range(len(img.shape)):
    # #     temp = ((img[:, :, ch].astype(float) - A[ch]) / t) + A[ch]
    # #     temp = np.maximum(np.minimum(temp, 255), 0)
    # #     HazeCorrectedImage[:, :, ch] = temp

    # print(path, '的时间', (consttime - start))
print( (end - start))
    # print("算A与tb %.3f" % (consttime - start))
    # print(end - start)
    # print(1 / (end - start))