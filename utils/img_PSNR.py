import os

import cv2
img_dir = os.listdir('../img')
print(img_dir)
for oneimg in img_dir:
    path1 = '../img/' + oneimg
    path2 = '../out/out_My/' +oneimg
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    jisuanpsnr = cv2.PSNR(img1, img2)
    print(oneimg,jisuanpsnr)