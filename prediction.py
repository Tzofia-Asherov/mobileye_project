import numpy as np
from PIL import Image

def crop(img, cord):
        # get np.array return image
        img = np.array(img)
        height, width, _ = img.shape
        left = cord[0] - 40
        top = cord[1] - 40
        right = cord[0] + 41
        bottom = cord[1] + 41
        return np.array(Image.fromarray(img).crop((left, top, right, bottom)))
