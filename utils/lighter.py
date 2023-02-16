import cv2
import numpy as np


def img_contrast_bright(img):
    a = 1.3
    b = 1 - a
    g = 10
    h,w,c = img.shape
    blank = np.zeros([h,w,c],img.dtype)
    dst = cv2.addWeighted(img,a,blank,b,g)
    return dst



path ='../out/dji_verybig.jpg'
img = cv2.imread(path)
dstimg=img_contrast_bright(img)
cv2.imwrite('../out/dji_verybig_lighter.jpg', dstimg)
cv2.imshow('1', dstimg)
cv2.waitKey(0)