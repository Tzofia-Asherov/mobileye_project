import numpy as np
import matplotlib.pyplot as plt


# func that create a circle filter and return it. It get a size of  filter and the radios
def circleFilter(len_filter, r=4):
    cir = np.zeros([len_filter, len_filter])
    for i in range(0, len_filter):
        for j in range(0, len_filter):
            # if: in the circle the color is black
            if (int(len_filter / 2) - i) ** 2 + (int(len_filter / 2) - j) ** 2 <= r ** 2:
                cir[i, j] = 1
            # else: In the scope to gray color
            elif ((int(len_filter / 2) - i) ** 2 + (int(len_filter / 2) - j) ** 2) < (r + 1) ** 2:
                cir[i, j] = 0.5
    return cir


