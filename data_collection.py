class DataCollection:
    def __init__(self):
        self.current_image = None
        self.prev_image = None
        self.em = None
        self.prev_points = []
        self.current_points = []

    def init_image_em(self,image, em,):
        self.current_image=image
        self.em = em

    # red_list,green_list)
    def init_part_one(self,x_red, y_red, x_green, y_green):
        self.x_red = x_red
        self.y_red = y_red
        self.x_green = x_green
        self.y_green  = y_green
        # self.red_list=red_list
        # self.green_list=green_list

    def init_part_two(self, predict_points):
        self.current_points = predict_points

    

