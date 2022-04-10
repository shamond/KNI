import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_lines(images, point_1, point_2, color=(255, 255, 255)):
    # points must be tuples
    return cv2.line(images, point_1, point_2, color)


def crop_image(images, convert=True):
    height, weight, channel = images.shape

    ROI = [(30, 480), (160, 80),
           (160, 80), (480, 80),
           (480, 80), (610, 480),
           #(670, 480), (800, 80),
           #(800, 80), (1120, 80),
           #(1120, 80), (1250, 480)
           ]

    mask = np.zeros_like(images)
    math_mask = (255,) * channel
    cv2.fillPoly(mask,
                 np.array([ROI], np.int32, ),
                 math_mask)

    masked_image = cv2.bitwise_and(images, mask)
    if not convert:
        return masked_image
    else:
        return cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)


def edges(images):
    return cv2.Canny(images, 100, 200)


def mean_filter(image, matrix):
    return cv2.blur(image, matrix)


def median_filter(image, reduce_precentage):
    return cv2.medianBlur(image, reduce_precentage)


def gauss_filter(images, matrix, sigma):
    # blurring the image with a 5x5, sigma = 1 Guassian kernel
    # matrix must be tuple matrix = (5,5) <=> 5x5
    return cv2.GaussianBlur(images, matrix, sigma)


def sharpness(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def brightness(image, alpha, beta):
    return cv2.addWeighted(image, alpha, np.zeros_like(image), 0, beta)
    # this method doing new image bright or dark e.g [[5,15,12],
    #                                                [10,15,20],  <=> g(x) = alpha * f(x) + beta
    #                                                [25,70,30]]
    # while is alfa = 1 and beta is minus values you reciving dark image
    # if alpha= 2 and beta = 50 new matrix will has [[60,80,74],
    #                                               [70,80,90],
    #                                               [100,190,110]


# color_line must be tuple to set color e.g (0,255,0) - green
def transform_hough(image, edges, color_line):
    # probablistic detection lines. this method use less memory than HoughLines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=50)
    for points in lines:
        x1, y1, x2, y2 = points[0]
        cv2.line(image, (x1, y1), (x2, y2), color_line)
    return image

#kontrast k = (Imax - imin) / (Imax + Imin)





def show(name, image):
    cv2.imshow(f"{name}", image)
    cv2.waitKey(1)


def histogram_color(image):
    # fig, ax = plt.subplots()
    # w,h,ch =image.shape
    # numpix = np.prod(image.shape)
    b, g, r = cv2.split(image)
    # lineB, = ax.plot(np.arange(256),np.zeros((256,)), c = "b" , alpha = .5, label = "blue")
    # lineR, = ax.plot(np.arange(256), np.zeros((256,)), c='r', alpha=.5, label='Red')
    # lineG, = ax.plot(np.arange(256), np.zeros((256,)), c='g', alpha=.5 , label='Green')
    # histogramR = cv2.calcHist([r], [0], None, [256], [0, 256]) / numpix
    # histogramG = cv2.calcHist([g], [0], None, [256], [0, 256]) / numpix
    # histogramB = cv2.calcHist([b], [0], None, [256], [0, 256]) / numpix
    # # cv2.normalize(histogramB, histogramB, alpha=0, beta=480, norm_type=cv2.NORM_MINMAX)
    # # cv2.normalize(histogramG, histogramG, alpha=0, beta=480, norm_type=cv2.NORM_MINMAX)
    # # cv2.normalize(histogramR, histogramR, alpha=0, beta=480, norm_type=cv2.NORM_MINMAX)
    #
    # lineR.set_ydata(histogramR)
    # lineG.set_ydata(histogramG)
    # lineB.set_ydata(histogramB)
    # ax.set_xlim(0, 256 - 1)
    # ax.set_ylim(0, 1)
    #
    #
    # #plt.plot(histogramB, histogramG, histogramR)
    # fig.canvas.draw()
    # plt.ion()

    plt.hist(b.ravel(), 256, [0, 256], density=True)
    plt.hist(g.ravel(), 256, [0, 256], density=True)
    plt.hist(r.ravel(), 256, [0, 256], density=True)

    return plt.show()


def histogram_gray(image):
    hist_img = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist_img, hist_img, alpha=0, beta=480, norm_type=cv2.NORM_MINMAX)
    plt.plot(hist_img)
    return plt.show()


