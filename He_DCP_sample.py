import time

import cv2
import numpy
def show(img):
    cv2.imshow('result', img)
    cv2.waitKey(0)  # 按任意键继续执行，可以自定义设置时间，单位毫秒
    # cv2.destroyAllWindows()
def darkchannel(imgar, ps=15):
    """
    Dark Channel estimation. According to equation (5) in the reference paper
    http://research.microsoft.com/en-us/um/people/kahe/cvpr09/

    Parameters
    -----------
    imgar:   an H*W RGB  hazed image
    ps:      the patch size (a patch P(x) has size (ps x ps) and is centered at pixal x)

    Return
    -----------
    The dark channel estimeted in imgar, jdark (a matrix H*W).
    """

    # Padding of the image to have windows of ps x ps size centered at each image pixel
    impad = numpy.pad(imgar, [(ps / 2, ps / 2), (ps / 2, ps / 2), (0, 0)], 'edge')

    # Jdark is the Dark channel to be found
    jdark = numpy.zeros((imgar.shape[0], imgar.shape[1]))

    for i in range(ps / 2, (imgar.shape[0] + ps / 2)):
        for j in range(ps / 2, (imgar.shape[1] + ps / 2)):
            # creates the patch P(x) of size ps x ps centered at x
            patch = impad[i - ps / 2:i + 1 + ps / 2, j - ps / 2:j + 1 + ps / 2]
            # selects the minimum value in this patch and set as the dark channel of pixel x
            jdark[i - ps / 2, j - ps / 2] = patch.min()

    return jdark

def DarkChannel(im, sz=15):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def Cal_A(imgar, jdark, px=1e-3):
    """
    Automatic atmospheric light estimation. According to section (4.4) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imgar:    an H*W RGB hazed image
    jdark:    the dark channel of imgar
    px:       the percentage of brigther pixels to be considered (default=1e-3, i.e. 0.1%)

    Return
    -----------
    The atmosphere light estimated in imgar, A (a RGB vector).
    """

    # reshape both matrix to get it in 1-D array shape
    imgavec = numpy.resize(imgar, (imgar.shape[0 ] *imgar.shape[1], imgar.shape[2]))
    jdarkvec = numpy.reshape(jdark, jdark.size)

    # the number of pixels to be considered
    numpx = numpy.int(jdark.size * px)

    # index sort the jdark channel in descending order
    isjd = numpy.argsort(-jdarkvec)

    asum = numpy.array([0.0 ,0.0 ,0.0])
    for i in range(0, numpx):
        asum[:] += imgavec[isjd[i], :]

    A = numpy.array([0.0 ,0.0 ,0.0])
    A[:] = asum[: ] /numpx

    # returns the calculated airlight A
    return A

def guided_filter(imgar, p, r=40, eps=1e-3):
    """
    Filter refinement under the guidance of an image. O(N) implementation.
    According to the reference paper http://research.microsoft.com/en-us/um/people/kahe/eccv10/

    Parameters
    -----------
    imgar:    an H*W RGB image used as guidance.
    p:        the H*W filter to be guided
    r:        the radius of the guided filter (in pixels, default=40)
    eps:      the epsilon parameter (default=1e-3)

    Return
    -----------
    The guided filter p'.
    """
    # H: height, W: width, C:colors
    H, W, C = imgar.shape
    # S is a matrix with the sizes of each local patch (window wk)
    S = __boxfilter__(numpy.ones((H, W)), r)

    # the mean value of each channel in imgar
    mean_i = numpy.zeros((C, H, W))

    for c in range(0, C):
        mean_i[c] = __boxfilter__(imgar[:, :, c], r) / S

    # the mean of the guided filter p
    mean_p = __boxfilter__(p, r) / S

    # the correlation of (imgar, p) corr_ip in each channel
    mean_ip = numpy.zeros((C, H, W))
    for c in range(0, C):
        mean_ip[c] = __boxfilter__(imgar[:, :, c] * p, r) / S

    # covariance of (imgar, p) cov_ip in each channel
    cov_ip = numpy.zeros((C, H, W))
    for c in range(0, C):
        cov_ip[c] = mean_ip[c] - mean_i[c] * mean_p

    # variance of imgar in each local patch (window wk), used to build the matrix sigma_k in eq.(14)
    # The variance in each window is a 3x3 symmetric matrix with variance as its values:
    #           rr, rg, rb
    #   sigma = rg, gg, gb
    #           rb, gb, bb
    var_i = numpy.zeros((C, C, H, W))
    # variance of (Red, Red)
    var_i[0, 0] = __boxfilter__(imgar[:, :, 0] * imgar[:, :, 0], r) / S - mean_i[0] * mean_i[0]
    # variance of (Red, Green)
    var_i[0, 1] = __boxfilter__(imgar[:, :, 0] * imgar[:, :, 1], r) / S - mean_i[0] * mean_i[1]
    # variance of (Red, Blue)
    var_i[0, 2] = __boxfilter__(imgar[:, :, 0] * imgar[:, :, 2], r) / S - mean_i[0] * mean_i[2]
    # variance of (Green, Green)
    var_i[1, 1] = __boxfilter__(imgar[:, :, 1] * imgar[:, :, 1], r) / S - mean_i[1] * mean_i[1]
    # variance of (Green, Blue)
    var_i[1, 2] = __boxfilter__(imgar[:, :, 1] * imgar[:, :, 2], r) / S - mean_i[1] * mean_i[2]
    # variance of (Blue, Blue)
    var_i[2, 2] = __boxfilter__(imgar[:, :, 2] * imgar[:, :, 2], r) / S - mean_i[2] * mean_i[2]

    a = numpy.zeros((H, W, C))

    for i in range(0, H):
        for j in range(0, W):
            sigma = numpy.array([[var_i[0, 0, i, j], var_i[0, 1, i, j], var_i[0, 2, i, j]],
                                 [var_i[0, 1, i, j], var_i[1, 1, i, j], var_i[1, 2, i, j]],
                                 [var_i[0, 2, i, j], var_i[1, 2, i, j], var_i[2, 2, i, j]]])

            # covariance of (imgar, p) in pixel (i,j) for the 3 channels
            cov_ip_ij = numpy.array([cov_ip[0, i, j], cov_ip[1, i, j], cov_ip[2, i, j]])

            a[i, j] = numpy.dot(cov_ip_ij, numpy.linalg.inv(sigma + eps * numpy.identity(3)))  # eq.(14)

    b = mean_p - a[:, :, 0] * mean_i[0, :, :] - a[:, :, 1] * mean_i[1, :, :] - a[:, :, 2] * mean_i[2, :, :]  # eq.(15)

    # the filter p'  eq.(16)
    pp = (__boxfilter__(a[:, :, 0], r) * imgar[:, :, 0]
          + __boxfilter__(a[:, :, 1], r) * imgar[:, :, 1]
          + __boxfilter__(a[:, :, 2], r) * imgar[:, :, 2]
          + __boxfilter__(b, r)) / S

    return pp

def __boxfilter__(m, r):
    """
    Fast box filtering implementation, O(1) time.

    Parameters
    ----------
    m:  a 2-D matrix data normalized to [0.0, 1.0]
    r:  radius of the window considered

    Return
    -----------
    The filtered matrix m'.
    """
    # H: height, W: width
    H, W = m.shape
    # the output matrix m'
    mp = numpy.zeros(m.shape)

    # cumulative sum over y axis
    ysum = numpy.cumsum(m, axis=0)
    # copy the accumulated values of the windows in y
    mp[0:r + 1, :] = ysum[r:(2 * r) + 1, :]
    # differences in y axis
    mp[r + 1:H - r, :] = ysum[(2 * r) + 1:, :] - ysum[:H - (2 * r) - 1, :]
    mp[(-r):, :] = numpy.tile(ysum[-1, :], (r, 1)) - ysum[H - (2 * r) - 1:H - r - 1, :]

    # cumulative sum over x axis
    xsum = numpy.cumsum(mp, axis=1)
    # copy the accumulated values of the windows in x
    mp[:, 0:r + 1] = xsum[:, r:(2 * r) + 1]
    # difference over x axis
    mp[:, r + 1:W - r] = xsum[:, (2 * r) + 1:] - xsum[:, :W - (2 * r) - 1]
    mp[:, -r:] = numpy.tile(xsum[:, -1][:, None], (1, r)) - xsum[:, W - (2 * r) - 1:W - r - 1]

    return mp

def tramission(imgar, A, w=0.95):
    """
    Transmission estimation. According to section (4.1) equation (11) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imgar:    an H*W RGB hazed image
    A:        the atmospheric light of imgar
    w:        the omega weight parameter, the amount of haze to be removed (default=0.95)

    Return
    -----------
    The transmission estimated in imgar, t (a H*W matrix).
    """
    # the normalized haze image
    nimg = numpy.zeros(imgar.shape)

    # calculate the normalized haze image
    for c in range(0, imgar.shape[2]):
        nimg[:, :, c] = imgar[:, :, c] / A[c]

    # estimate the dark channel of the normalized haze image
    njdark = DarkChannel(nimg)

    # calculates the transmisson t
    t = 1 - w * njdark + 0.25

    return t

def recover(imgar, atm, t, tmin=0.1):
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
    j = numpy.zeros(imgar.shape)

    # equation (16)
    for c in range(0, imgar.shape[2]):
        j[:, :, c] = ((imgar[:, :, c] - atm[c]) / numpy.maximum(t[:, :], tmin)) + atm[c]

    return j / numpy.amax(j)



def dehaze(imgar, a=None, t=None, rt=None, tmin=0.1, ps=15, w=0.95, px=1e-3, r=40, eps=1e-3, m=False):
    """
    Application of the dehazing algorithm, as described in section (4) of the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imgar:    an H*W RGB hazed image
    a:        the atmospheric light RGB array of imgar (default=None, will be calculated internally)
    t:        the transmission matrix H*W of imgar (default=None, will be calculated internally)
    rt:       the raw transmission matrix H*W of imgar, to be refined (default=None, will be calculated internally)
    tmin:     the minimum value the transmission can take (default=0.1)
    ps:       the patch size for dark channel estimation (default=15)
    w:        the omega weight, ammount of haze to be kept (default=0.95)
    px:       the percentage of brightest pixels to be considered when estimating atmospheric light (default=1e-3, i.e. 0.1%)
    r:        the radius of the guided filter in pixels (default=40)
    eps:      the epsilon parameter for guided filter (default=1e-3)
    m:        print out messages along processing

    Return
    -----------
    The dehazed image version of imgar, dehazed (a H*W RGB matrix).
    """

    jdark = darkchannel(imgar, ps)
    # return jdark
    # if no atmospheric given
    if a == None:
        a = Cal_A(imgar, jdark)
        if (m == True):
            print('Atmospheric Light estimated.')

    # if no raw transmission and complete transmission given
    if rt == None and t == None:
        rt = tramission(imgar, a, w)
        # threshold of raw transmission
        rt = numpy.maximum(rt, tmin)
        if (m == True):
            print('Transmission estimated.')

    # if no complete transmission given, refine the raw using guided filter
    if t == None:
        t = guided_filter(imgar, rt)
        if (m == True):
            print('Transmission refined.')

    # recover the scene radiance
    dehazed = recover(imgar, a, t, tmin)
    if (m == True):
        print('Radiance recovered.')

    return dehazed

path = './imgHazeKS/4.png'
img = cv2.imread(path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
start = time.time()
ddark = DarkChannel(img, sz=15)
A = Cal_A(img, ddark)
t= tramission(img, A)

# out = Recover(img, t, A)
end = time.time()
out2 = recover(img,A,t)
# meng
# HazeCorrectedImage = copy.deepcopy(img)
# for ch in range(len(img.shape)):
#     temp = ((img[:, :, ch].astype(float) - A[ch]) / t) + A[ch]
#     temp = np.maximum(np.minimum(temp, 255), 0)
#     HazeCorrectedImage[:, :, ch] = temp
print(end-start)
print(1/(end-start))
show(out2)