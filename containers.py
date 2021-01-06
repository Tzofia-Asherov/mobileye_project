
class FrameContainer(object):
    def __init__(self, image):
        # reader = png.Reader(img_path)
        # np_img  = reader.read()
        # float_array = np.array( map( np.uint16, np_img[2] ) )
        # self.img = float_array.astype(int)
        self.img = image


        
        #self.img = png.read_png_int(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind=[]
        self.valid=[]