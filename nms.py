import numpy as np
import scipy.misc
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import imageio
from skimage.color import rgb2gray
from color import green_original, red_original, green_vectorized

# func that get image after convnvolution and the name of picture
def nms(greyImage, images, name):
    i = 0
    j = 0
    x_green, y_green = [], []
    x_red, y_red = [], []
    list_of_color = ["ro", "go", "bo"]
    for typ in images:
        neighborhood_size=0
        threshold=0
        if typ[1] == 4:
            neighborhood_size = 25
            threshold = 150
        if typ[1] == 0:
            neighborhood_size = 8  # could be 8
            threshold = 25

        data = typ[0]
        data_max = filters.maximum_filter(data, neighborhood_size)  # get the local nmax
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        np_img = np.array(greyImage)
        for dy, dx in slices:  # append point to list of points 1 of x points and 2 to y points
            x = typ[1]
            if len(np_img[dy.start:dy.stop, dx.start:dx.stop]):
                if green_original(np_img[dy.start - x:dy.stop + x, dx.start - x:dx.stop + x]) > 50:
                    i += 1
                    x_center = (dx.start + dx.stop - 1) / 2
                    x_green.append(x_center)
                    y_center = (dy.start + dy.stop - 1) / 2
                    y_green.append(y_center)
                if red_original(np_img[dy.start - x:dy.stop + x, dx.start - x:dx.stop + x]) > 15:
                    i += 1
                    x_center = (dx.start + dx.stop - 1) / 2
                    x_red.append(x_center)
                    y_center = (dy.start + dy.stop - 1) / 2
                    y_red.append(y_center)
        j += 1
    # print("i", i)
    # plt.clf()

    return x_red, y_red, x_green, y_green
