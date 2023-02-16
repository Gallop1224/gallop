
import cv2
import os

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import structural_similarity as SSIM

img_dir = os.listdir('../img')
print(img_dir)
for oneimg in img_dir:
    path1 = '../img/' + oneimg
    path2 = '../out/out_meng/' +oneimg
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)


    jisuanssim = SSIM(img1, img2)
    print(oneimg,jisuanssim)









def average_ssim(img1 ,img2):

    sr_dir = os.listdir('./SR')
    hr_dir = os.listdir('./HR')

    psnr = 0.0
    ssim = 0.0
    n = 0

    def to_grey(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for hr_image in hr_dir:
        for sr_image in sr_dir:
            if sr_image == hr_image:
                if (sr_image[-3:]) != 'png':
                    continue
                compute_psnr = cv2.PSNR(cv2.imread('./SR/' + sr_image), cv2.imread('./HR/' + hr_image))
                compute_ssim = SSIM(to_grey(cv2.imread('./SR/' + sr_image)),
                                            to_grey(cv2.imread('./HR/' + hr_image)))
                psnr += compute_psnr
                ssim += compute_ssim
                n += 1
                if n% 100 == 0:
                    print("finish compute [%d/%d]" % (n, len(hr_dir)))

    psnr = psnr / n
    ssim = ssim / n
    print("average psnr = ", psnr)
    print("average ssim = ", ssim)

