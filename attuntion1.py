try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")

# def create_kernel():
#     kernel = []
#     for i in range(21):
#         row = []
#         for j in range(21):
#             row.append(-0.25)
#         kernel.append(row)
#     for i in range(15):
#         for j in range(15):
#             kernel[i+3][j+3] = 0.24
#     return kernel

#treshold = 500
# def create_kernel():
#     kernel = []
#     for i in range(13):
#         row = []
#         for j in range(13):
#             row.append(-0.2)
#         kernel.append(row)
#     for i in range(9):
#         for j in range(9):
#             kernel[i+2][j+2] = 0.2
#     kernel[6][6] += 1.4
#     return kernel

#treshold 300 good!! no catch big and not catch very very small
# def create_kernel():
#      kernel = []
#      for i in range(17):
#          row = []
#          for j in range(17):
#              row.append(-0.4)
#          kernel.append(row)
#      for i in range(9):
#          for j in range(9):
#              kernel[i+4][j+4] = 0.5
#      kernel[8][8] += 6.7
#      return kernel


def create_kernel():
    kernel = []
    for i in range(15):
        row = []
        for j in range(15):
            row.append(-0.2)
        kernel.append(row)
    for i in range(9):
        for j in range(9):
            kernel[i+3][j+3] = 0.34
    kernel[7][7] += 0.66
    return kernel

# 2 frame 500
# def create_kernel():
#     kernel = []
#     for i in range(13):
#         row = []
#         for j in range(13):
#             row.append(-0.2)
#         kernel.append(row)
#     for i in range(9):
#         for j in range(9):
#             kernel[i+2][j+2] = 0.2
#     kernel[6][6] += 1.4
#     return kernel
# #

def get_lights_indexes(data_max):
    x = ()
    y = ()
    size = data_max.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if data_max[i][j]:
                x = x + (j,)
                y = y + (i,)
    return x, y

def calc_if_small_distance(x, y, tuple_x, tuple_y):
    current_point = np.array((x, y))
    for j in range(len(tuple_x)):
        index_x = tuple_x[j]
        index_y = tuple_y[j]
        point = np.array((index_x, index_y))
        if x != index_x or y != index_y:
            distance = np.linalg.norm(current_point - point)
            if (x == index_x and y != index_y) or (y == index_y and x != index_x):
                if distance <= 30:
                    return True
            else:
                if distance <= 10:
                    return True
    return False
# def general_calc_if_small_distance(x, y, tuple_x, tuple_y):
#     current_point = np.array((x, y))
#     for j in range(len(tuple_x)):
#         index_x = tuple_x[j]
#         index_y = tuple_y[j]
#         point = np.array((index_x, index_y))
#         if x != index_x or y != index_y:
#             distance = np.linalg.norm(current_point - point)
#             if distance <= 30:
#                 return True
#     return False
#
# def calc_if_small_distance(x, y, tuple_x, tuple_y):
#     current_point = np.array((x, y))
#     for j in range(len(tuple_x)):
#         index_x = tuple_x[j]
#         index_y = tuple_y[j]
#         point = np.array((index_x, index_y))
#         if (x == index_x and y != index_y) or (y == index_y and x != index_x):
#             distance = np.linalg.norm(current_point - point)
#             if distance <= 13:
#                 return True
#     return False

def seperate_color(tuple_x, tuple_y, src_image):
    x_red = ()
    y_red = ()
    x_green = ()
    y_green = ()

    points = zip(tuple_x, tuple_y)
    for x, y in points:
        red_val = src_image[y, x, 0]
        green_val = src_image[y, x, 1]
        if red_val >= green_val:
            x_red += (x,)
            y_red += (y,)
        else:
            x_green += (x,)
            y_green += (y,)
    return x_red, y_red, x_green, y_green



def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green

    """
    src_image = c_image
    c_image = np.dot(c_image[..., :3], [0.299, 0.587, 0.114])

    kernel = create_kernel()

    #c_image =  ndimage.gaussian_filter(c_image, sigma=1)
    mat_konvulutzia =  ndimage.convolve(c_image, kernel, mode='constant', cval=0.0)

    #plt.imshow(mat_konvulutzia, cmap="gray")

    data_max = maximum_filter(mat_konvulutzia, size=5)
    data_max[data_max <= kwargs['some_threshold']] = 0

    data_max[data_max != mat_konvulutzia] = 0
    data_max[0:22, :] = 0
    data_max[-22:-1, :] = 0
    data_max[:, 0:22] = 0
    data_max[:, -22:-1] = 0


    tuple_x, tuple_y = get_lights_indexes(data_max)

    tuple_x_zip = ()
    tuple_y_zip = ()
    for i in range(len(tuple_x)):
        current_index_x = tuple_x[i]
        current_index_y = tuple_y[i]
        if not calc_if_small_distance(current_index_x, current_index_y,tuple_x, tuple_y):
                tuple_x_zip += (current_index_x, )
                tuple_y_zip += (current_index_y, )

    x_red, y_red, x_green, y_green = seperate_color(tuple_x_zip, tuple_y_zip, src_image)

    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    # plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=1500)
    plt.plot(red_x, red_y, 'r*', color='r', markersize=4)
    plt.plot(green_x, green_y, 'r*', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = './images'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    #plt.show(block=True)


if __name__ == '__main__':
    main()
