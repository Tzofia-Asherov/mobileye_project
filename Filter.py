import matplotlib.pyplot as plt
import numpy as np
# import scipy
from scipy import ndimage
from PIL import Image
# import cv2
from nms import nms
from circles import circleFilter
import os


# func that plot subplt of 4 figure of gray pictures
def plot(data, title):
    plot.i += 1
    plt.subplot(2, 2, plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)


plot.i = 0



# Load the data and convert rgb to gray image.
def covulotion_images(images):
    image_index = 0
    highpass_kernel = np.array([[-1, -1, -1, -1, -1],
                                [-1, 1, 2, 1, -1],
                                [-1, 2, 4, 2, -1],
                                [-1, 1, 2, 1, -1],
                                [-1, -1, -1, -1, -1]])
    lowpass_kernel = np.array([[1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
                               [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
                               [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
                               [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9],
                               [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9]])

    for img in images:
        im = img.copy().convert('1')
        np_image = np.array(im, dtype=float)
        '''first make high pass filter and after on it make low pass filter.'''
        # plt.imsave("greyscale.jpg", np_image)

        # high pass filter
        highpass_5x5 = ndimage.convolve(np_image, highpass_kernel)
        plt.imsave("highpass_5x5.jpg", highpass_5x5)


        # low pass filter
        highpass_5x5 = ndimage.convolve(highpass_5x5, lowpass_kernel)
        # plt.imsave("lowpass_5x5.jpg", highpass_5x5)

        '''create some of filter with other len and radius and after call to convolation func '''
        cir0 = circleFilter(3, 1)
        cir_image0 = ndimage.convolve(highpass_5x5, cir0)

        cir1 = circleFilter(10, 7)
        cir_image1 = ndimage.convolve(highpass_5x5, cir1)

        # plt.imsave("cir_image0.jpg", cir_image0)
        # plt.imsave("cir_image1.jpg", cir_image1)

        '''call to func non maximum supression with all convolution and the num of image'''
        image_index += 1
        return nms(img, [(cir_image0.copy(), 0), (cir_image1.copy(), 4)], str(image_index))
