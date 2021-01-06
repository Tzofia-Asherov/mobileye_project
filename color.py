import numpy as np


# Your original function, with the file i/o removed for timing comparison
def green_original(x):
    count_green = 0
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            pixel = list(map(int, x[i, j].tolist()))
            # if sum(pixel) != 0:
            if sum(pixel) > 400:
                green_pixel = 100 * (pixel[1] / sum(pixel))
                blue_pixel = 100 * (pixel[2] / sum(pixel))
                red_pixel = 100 * (pixel[0] / sum(pixel))
                if green_pixel >= red_pixel and green_pixel >= blue_pixel:
                    if green_pixel > 37:
                        count_green += 1
    if x.shape[0] * x.shape[1] == 0:
        return 0
    green_percent = round(100 * (count_green / (x.shape[0] * x.shape[1])), 2)
    return green_percent
def red_original(x):
    #return 50
    count_red = 0
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            pixel = list(map(int, x[i, j].tolist()))
            # if sum(pixel) != 0:
            if sum(pixel) > 300:
                green_pixel = 100 * (pixel[1] / sum(pixel))
                blue_pixel = 100 * (pixel[2] / sum(pixel))
                red_pixel = 100 * (pixel[0] / sum(pixel))
                if  red_pixel>=green_pixel  and red_pixel >= blue_pixel:
                    if red_pixel > 25:
                        count_red += 1
    if x.shape[0] * x.shape[1] == 0:
        return 0
    red_percent = round(100 * (count_red / (x.shape[0] * x.shape[1])), 2)
    return red_percent


def green_vectorized(x):
    mask = (img[:, :, 1] > img[:, :, 0]) & (img[:, :, 1] > img[:, :, 2]) & ((img[:, :, 1] / np.sum(img, axis=2)) > .35)
    return round(100 * np.sum(mask) / (x.shape[0] * x.shape[1]), 2)


img = np.ones(shape=(300, 300, 3))
# img[0:150, 0:150, 1] = 134
# print(green_original(img[0:150, 0:150]))

# %timeit green_vectorized(img)
