import math

import cv2
import numpy as np

# img = '../img/4.png'
# img = cv2.imread(img)
# im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # pix = min(im.shape)
# # n = math.log2(pix/windowsize)
# # print(n)
# # n = int(n)
# # print(n)
# imlist = [[], []]
# for times in range(3):
#     h, w = im.shape[0], im.shape[1]
#     imlist[0].append(im[0:int(w / 2), 0:int(h / 2)])
#     imlist[0].append(im[int(w / 2):w, 0:int(h / 2)])
#     imlist[0].append(im[0:int(h / 2), int(h / 2):h])
#     imlist[0].append(im[int(w / 2):w, int(h / 2):h])
#     imlist[1].append(img[0:int(w / 2), 0:int(h / 2)])
#     imlist[1].append(img[int(w / 2):w, 0:int(h / 2)])
#     imlist[1].append(img[0:int(h / 2), int(h / 2):h])
#     imlist[1].append(img[int(w / 2):w, int(h / 2):h])
#     scorelist = [np.mean(i) for i in imlist[0]]
#     max_index = np.argmax(scorelist)
#     im, img = imlist[0][max_index], imlist[1][max_index]
#     imlist = [[], []]
# img_single_channel_list = cv2.split(img)
# A = []
# for channel in img_single_channel_list:
#     A.append(np.mean(channel))

def img_contrast_bright(img):
    a = 1.3
    b = 1 - a
    g = 10
    h, w, c = img.shape
    blank = np.zeros([h, w, c], img.dtype)
    dst = cv2.addWeighted(img, a, blank, b, g)
    return dst
def show(img):
    cv2.imshow('result', img)
    cv2.waitKey(0)  # 按任意键继续执行，可以自定义设置时间，单位毫秒
    # cv2.destroyAllWindows()
img2 = '../out/out_meng/DJI_0032.00_00_41_11.Still001.jpg'
img2 = cv2.imread(img2)
out = img_contrast_bright(img2)
show(out)
cv2.imwrite('../out/out_meng/test1.jpg', out)