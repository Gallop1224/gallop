import copy
import os
import sys
import time

import cv2
import numpy as np


# 估算大气光A
def Airlight(HazeImg, AirlightMethod, windowSize):
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



# 计算边界限制tb(x)
# tb(x)=min{ max( (A-I(X))/(A-C0),(A-I(X))/(A-C1) ) , 1 }
def BoundaryConstraints(HazeImg, A, C0, C1, windowSze):
    if len(HazeImg.shape) == 3:
        t_b = np.maximum((A[0] - HazeImg[:, :, 0].astype(np.float64)) / (A[0] - C0),  # np.maximum(X, Y) 用于逐元素比较两个array的大小。
                         (HazeImg[:, :, 0].astype(np.float64) - A[0]) / (C1 - A[0]))  # max( (A-I(X))/(A-C0),
        t_g = np.maximum((A[1] - HazeImg[:, :, 1].astype(np.float64)) / (A[1] - C0),  #      (A-I(X))/(A-C1) )
                         (HazeImg[:, :, 1].astype(np.float64) - A[1]) / (C1 - A[1]))
        t_r = np.maximum((A[2] - HazeImg[:, :, 2].astype(np.float64)) / (A[2] - C0),
                         (HazeImg[:, :, 2].astype(np.float64) - A[2]) / (C1 - A[2]))

        MaxVal = np.maximum(t_b, t_g, t_r)
        transmission = np.minimum(MaxVal, 1)                                          # min{ max( (A-I(X))/(A-C0),(A-I(X))/(A-C1) ) , 1 }
    else:  # 单通道
        transmission = np.maximum((A[0] - HazeImg.astype(np.float64)) / (A[0] - C0),
                                  (HazeImg.astype(np.float64) - A[0]) / (C1 - A[0]))
        transmission = np.minimum(transmission, 1)

    kernel = np.ones((windowSze, windowSze), np.float64)
    transmission2_tbx = cv2.morphologyEx(transmission, cv2.MORPH_CLOSE, kernel=kernel)   #  先进行膨胀操作，再进行腐蚀操作
    return transmission2_tbx                                                             # 可以填充小的黑洞（fill hole补洞），去掉小的黑噪点。填充目标区域内的离散小空洞和分散部分

'''
开始计算传输率t(x)
'''

# 滤波算子
def LoadFilterBank():
    KirschFilters = []
    KirschFilters.append(np.array([[-3, -3, -3],   [-3, 0, 5],   [-3, 5, 5]]))
    KirschFilters.append(np.array([[-3, -3, -3],   [-3, 0, -3],  [5, 5, 5]]))
    KirschFilters.append(np.array([[-3, -3, -3],   [5, 0, -3],   [5, 5, -3]]))
    KirschFilters.append(np.array([[5, -3, -3],    [5, 0, -3],   [5, -3, -3]]))
    KirschFilters.append(np.array([[5, 5, -3],     [5, 0, -3],   [-3, -3, -3]]))
    KirschFilters.append(np.array([[5, 5, 5],      [-3, 0, -3],  [-3, -3, -3]]))
    KirschFilters.append(np.array([[-3, 5, 5],     [-3, 0, 5],   [-3, -3, -3]]))
    KirschFilters.append(np.array([[-3, -3, 5],    [-3, 0, 5],   [-3, -3, 5]]))
    KirschFilters.append(np.array([[-1, -1, -1],   [-1, 8, -1],  [-1, -1, -1]]))
    return(KirschFilters)
# # return---wj(i)
def CalculateWeightingFunction(HazeImg, Filter, sigma):
    # Computing the weight function... Eq (17) in the paper
    # wj(i) =e~((-sum(dj COV I)~2)/2*SIGMA*SIGMA)
    # 基于两个相邻像素的颜色向量之间的平方差
    HazeImageDouble = HazeImg.astype(float) / 255.0
    if len(HazeImg.shape) == 3:
        Red = HazeImageDouble[:, :, 2]
        d_r = circularConvFilt(Red, Filter)

        Green = HazeImageDouble[:, :, 1]
        d_g = circularConvFilt(Green, Filter)

        Blue = HazeImageDouble[:, :, 0]
        d_b = circularConvFilt(Blue, Filter)

        WFun = np.exp(-((d_r ** 2) + (d_g ** 2) + (d_b ** 2)) / (2 * sigma * sigma))
    else:
        d = circularConvFilt(HazeImageDouble, Filter)
        WFun = np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * sigma * sigma))
    return WFun
#
#
# # 卷积计算
def circularConvFilt(Img, Filter):
    FilterHeight, FilterWidth = Filter.shape
    assert (FilterHeight == FilterWidth), 'Filter must be square in shape --> Height must be same as width'
    assert (FilterHeight % 2 == 1), 'Filter dimension must be a odd number.'
    # cv2.BORDER_WRAP这种边界填充在filter2d中不支持，所以要单独填充边界
    filterHalsSize = int((FilterHeight - 1) / 2)
    rows, cols = Img.shape
    PaddedImg = cv2.copyMakeBorder(Img, filterHalsSize, filterHalsSize, filterHalsSize, filterHalsSize,
                                   borderType=cv2.BORDER_WRAP)
    FilteredImg = cv2.filter2D(PaddedImg, -1, Filter)
    Result = FilteredImg[filterHalsSize:rows + filterHalsSize, filterHalsSize:cols + filterHalsSize]  #减去刚刚扩充的边界

    return (Result)
#
# '''
# Filter = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
# img = cv2.imread('fishers.jpg',cv2.IMREAD_GRAYSCALE)
# out  = circularConvFilt(img,Filter)
# print(out)
# '''
#
#
def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img
#
#
def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf
#
#
# # 通过固定uj或t来最小化目标函数，找到最优tx
# # return---tx
def CalTransmission(HazeImg, Transmission, regularize_lambda, sigma):
    rows, cols = Transmission.shape
    time1 = time.time()
    KirschFilters = LoadFilterBank()

    # Normalize the filters
    for idx, currentFilter in enumerate(KirschFilters):
        KirschFilters[idx] = KirschFilters[idx] / np.linalg.norm(currentFilter)

    # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
    WFun = []
    for idx, currentFilter in enumerate(KirschFilters):
        WFun.append(CalculateWeightingFunction(HazeImg, currentFilter, sigma))

    # Precompute the constants that are later needed in the optimization step
    tF = np.fft.fft2(Transmission)
    DS = 0

    for i in range(len(KirschFilters)):
        D = psf2otf(KirschFilters[i], (rows, cols))
        DS = DS + (abs(D) ** 2)

    # Cyclic loop for refining t and u --> Section III in the paper
    beta = 1  # Start Beta value --> selected from the paper
    beta_max = 2 ** 8  # Selected from the paper --> Section III --> "Scene Transmission Estimation"
    beta_rate = 2 * np.sqrt(2)  # Selected from the paper

    while (beta < beta_max):
        gamma = regularize_lambda / beta

        # Fixing t first and solving for u
        DU = 0
        for i in range(len(KirschFilters)):
            dt = circularConvFilt(Transmission, KirschFilters[i])
            u = np.maximum((abs(dt) - (WFun[i] / (len(KirschFilters) * beta))), 0) * np.sign(dt)
            DU = DU + np.fft.fft2(circularConvFilt(u, cv2.flip(KirschFilters[i], -1)))

        # Fixing u and solving t --> Equation 26 in the paper
        # Note: In equation 26, the Numerator is the "DU" calculated in the above part of the code
        # In the equation 26, the Denominator is the DS which was computed as a constant in the above code

        Transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
        beta = beta * beta_rate

    time2=time.time()
    time3= time2-time1
    return (Transmission,time3)




def removeHaze_singleImg(HazeImg,  delta=0.65):
    '''
    :param HazeImg: Hazy input image
    :param Transmission: estimated transmission
    :param A: estimated airlight
    :param delta: fineTuning parameter for dehazing --> default = 0.85
    :return: result --> Dehazed image
    '''

    # This function will implement equation(3) in the paper
    # " https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.pdf "

    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    calAtime1= time.time()
    A = Airlight(HazeImg, AirlightMethod, windowSze)
    calAtime2=time.time()
    calAtime=calAtime2-calAtime1
    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20  # Default value = 20 (as recommended in the paper)
    C1 = 300  # Default value = 300 (as recommended in the paper)
    calTbtime1 = time.time()
    Transmission_tbx = BoundaryConstraints(HazeImg, A, C0, C1,
                                windowSze)  # Computing the Transmission using equation (7) in the paper
    calTbtime2 = time.time()
    calTbtime = calTbtime2-calTbtime1

    # Refine estimate of transmission
    regularize_lambda = 1  # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    sigma = 0.5
    Transmission_tx, calTtime = CalTransmission(HazeImg, Transmission_tbx, regularize_lambda,
                                      sigma)  # Using contextual information


    # print(Transmission_tx)


    epsilon = 0.0001
    Transmission_tx = pow(np.maximum(abs(Transmission_tx), epsilon), delta)

    HazeCorrectedImage = copy.deepcopy(HazeImg)
    if(len(HazeImg.shape) == 3):
        for ch in range(len(HazeImg.shape)):
            temp = ((HazeImg[:, :, ch].astype(float) - A[ch]) / Transmission_tx) + A[ch]
            temp = np.maximum(np.minimum(temp, 255), 0)
            HazeCorrectedImage[:, :, ch] = temp
    else:
        temp = ((HazeImg.astype(float) - A[0]) / Transmission_tx) + A[0]
        temp = np.maximum(np.minimum(temp, 255), 0)
        HazeCorrectedImage = temp

    # return HazeCorrectedImage
    return HazeCorrectedImage, calTtime, calAtime,calTbtime
def show(img):
    cv2.imshow('result', img)
    cv2.waitKey(0)





img_dir = os.listdir('./imgHazeKS')
print(img_dir)
for oneimg in img_dir:
    path = './imgHazeKS/' + oneimg

    start = time.time()
    img = cv2.imread(path)
    out, calTtime, calAtime, calTbtime = removeHaze_singleImg(img)
    cv2.imwrite('./outKS/out_meng/' + oneimg, out)
    end = time.time()
    #print(oneimg,end-start)
    print( end - start)

    # print(calAtime)
    # print(calTbtime)
    # print(calTtime)
    # show(out)

