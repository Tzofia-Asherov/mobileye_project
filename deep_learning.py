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

    import run_attention
    import random

except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")


def crop_img_by_center(img, cord):
    # get np.array return image
    height, width, _ = img.shape
    left = cord[1] - 40
    top = cord[0] - 40
    right = cord[1] + 41
    bottom = cord[0] + 41
    return np.array(Image.fromarray(img).crop((left, top, right, bottom)))


# def get_crop_image(x, y, real_image):
#     y_begin = y - 40
#     y_end = y + 41
#     x_begin = x - 40
#     x_end = x + 41
#     if y_begin < 0:
#         y_end += (y_begin * -1)
#         y_begin = 0
#     elif y_end > real_image.shape[0]:
#         y_begin -= y_end - real_image.shape[0]
#         y_end = real_image.shape[0]
#     if x_begin < 0:
#          x_end += (x_begin * -1)
#          x_begin = 0
#     elif x_end > real_image.shape[1]:
#          x_begin -= x_end - real_image.shape[1]
#          x_end = real_image.shape[1]
#     crop_image =  real_image[y_begin:y_end , x_begin:x_end , :]
#     return crop_image

def get_x_y_random(label_image, x_list, y_list):
    var = 19
    counter = 0
    while(var == 19):
        index = random.randint(0, len(x_list)-1)
        i = y_list[index]
        j = x_list[index]
        var = label_image[i, j]
        counter = counter + 1
        if counter == len(x_list):
            j = random.randint(0, label_image.shape[1])
            i = random.randint(0, label_image.shape[0])
            break
    return i, j


def build_dataset(label_list, path, json_fn):
    data = []
    labels =[]
    flag = False
    for label_path in label_list:
        try:
            if flag:
                label_path = label_path[1:]
            else:
                flag = True
            image_path = get_real_path_bcy_path_label(label_path)

            label_image = np.array(Image.open(label_path))
            real_image = np.array(Image.open(image_path))
            plt.imshow(real_image)
            plt.imshow(label_image)

            red_x, red_y, green_x, green_y = run_attention.test_find_tfl_lights(image_path, json_fn)
            # plt.imshow(block=True)

            x_list = red_x + green_x
            y_list = red_y + green_y


            for index in range(len(x_list)):
                j = x_list[index]
                i = y_list[index]
                if label_image[i][j] == 19:
                # print(label_image[i][j])
                # print(label_image[i, j])
                # if label_image[i, j] == 19:
                    crop_traffic_img = crop_img_by_center(real_image, (i, j))
                    # image = np.array(Image.open(crop_traffic_img))
                    # plt.imshow(crop_traffic_img)
                    data.append(crop_traffic_img)
                    labels.append(1)


                    i_rand, j_rand = get_x_y_random(label_image, x_list, y_list)
                    # print(label_image[i_rand])
                    crop_non_traffic_img = crop_img_by_center(real_image, (i_rand, j_rand))
                    # plt.imshow(crop_non_traffic_img)
                    data.append(crop_non_traffic_img)
                    labels.append(0)

                    #if len(data)%100 == 0:
                    print(len(data))

        except:
            print("error:" + label_path)

    data_path = path + 'data.bin'
    labels_path = path + 'labels.bin'
    np.array(data).tofile(data_path)
    np.array(labels).astype('uint8').tofile(labels_path)



def get_real_path_bcy_path_label(path_label):
    path_label = path_label.replace('gtFine_trainvaltest/gtFine', 'leftImg8bit_trainvaltest\leftImg8bit')
    path_label = path_label.replace('gtFine_labelIds', 'leftImg8bit')
    return path_label


def map_data(path, cities_list, file_name):
    traffic_lights_path = []
    file_path ='./label_data/' +file_name +'.txt'

    for i in range(len(cities_list)):
        city_path = path + cities_list[i]
        all_images_city_path = glob.glob(os.path.join(city_path, '*labelIds.png'))
        for image_path in all_images_city_path:
            image = np.array(Image.open(image_path))
            if 19 in image:
                traffic_lights_path.append(image_path)
    with open(file_path, 'w') as f:
        f.write("%s\n" % traffic_lights_path)



def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    # parser = argparse.ArgumentParser("Test TFL attention mechanism")
    # parser.add_argument('-i', '--image', type=str, help='Path to an image')
    # parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    # parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    # args = parser.parse_args(argv)

    # default_base = './images'

    # label_path_train = './gtFine_trainvaltest/gtFine/train/'
    # list_lable_train = ['aachen', 'bochum','bremen','cologne','darmstadt','dusseldorf','erfurt','hamburg','hanover','jena','krefeld',
    #                     'monchengladbach','strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
    # label_path_val = './gtFine_trainvaltest/gtFine/val/'
    # list_lable_val = ['frankfurt', 'lindau', 'munster']

    # map_data(label_path_train, list_lable_train, 'train_data')
    # map_data(label_path_val, list_lable_val, 'val_data')


    text_file_train = open("./label_data/train_data.txt", "r")
    label_list_train = text_file_train.readlines()[0][1:-1].split(',')
    label_list_train = [x[1:-1] for x in label_list_train]

    text_file_val = open("./label_data/val_data.txt", "r")
    label_list_val = text_file_val.readlines()[0][1:-1].split(',')
    label_list_val = [x[1:-1] for x in label_list_val]

    #build_dataset(label_list_train,"./Data_dir/train/", None)
    build_dataset(label_list_val,"./Data_dir/val/", None)

    if len(label_list_train) and len(label_list_val):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")


if __name__ == '__main__':
    main()
