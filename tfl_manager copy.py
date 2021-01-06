import numpy as np
import matplotlib.pyplot as plt
import Filter
import data_collection
import tensorflow as tf
from tensorflow.keras.models import load_model
import seaborn as sbn
from prediction import crop
import SFM as SFM
import containers



class TFL_manager:
    def __init__(self, focal, pp, model_path):

        self.focal = focal
        self.pp = pp
        self.data_collection = data_collection.DataCollection()
        self.loaded_model = load_model(model_path)

        

    def run(self, img, em,):
        self.data_collection.init_image_em(img, em)

        self.attention()
        self.prediction()

        self.calc_distances()

    def attention(self):
        #x_red, y_red, x_green, y_green  = att.find_tfl_lights(np.array(img),  some_threshold=1500)  
        x_red, y_red, x_green, y_green  = Filter.covulotion_images([self.data_collection.current_image]) 

        self.data_collection.init_part_one(x_red, y_red, x_green, y_green)

    def prediction(self):
        red_list = [(self.data_collection.x_red[i], self.data_collection.y_red[i]) for i in range(len(self.data_collection.x_red))]
        green_list =   [(self.data_collection.x_green[i], self.data_collection.y_green[i]) for i in range(len(self.data_collection.x_green))]


        predict_red_points = self.get_prediction__points(red_list) 
        predict_green_points = self.get_prediction__points(green_list)
        self.data_collection.init_part_two(predict_red_points + predict_green_points )
 
        #self.data_collection.init_part_two(predict_red_points )
    
    
    def get_prediction__points(self, all_points):

        tfl_points_machine = []
        val_images = [np.array(crop(self.data_collection.current_image ,point)) for point in all_points]
        

        crop_shape=(81,81)
        for i in range(len(val_images)):
            
            if val_images[i].shape == (81,81,3):
                img =val_images[i].reshape([-1]+list(crop_shape) +[3])
            
                predictions = self.loaded_model.predict(img)
                predicted_label = np.argmax(predictions, axis=-1)

                if predicted_label[0]:
                    tfl_points_machine.append((all_points[i][0], all_points[i][1]))
            
        return tfl_points_machine


    def calc_distances(self):
        if self.data_collection.em!= None:
            prev_container = containers.FrameContainer(self.data_collection.prev_image)
            curr_container = containers.FrameContainer(self.data_collection.current_image)
            curr_container.EM = self.data_collection.em[1]
            prev_container.traffic_light = np.array(self.data_collection.prev_points)
            curr_container.traffic_light = np.array(self.data_collection.current_points)
            curr_container = SFM.calc_TFL_dist(prev_container, curr_container, self.focal, self.pp)


        self.data_collection.prev_points = self.data_collection.current_points
        self.data_collection.prev_image = self.data_collection.current_image
        
        


    def visualization(self, prev_container, curr_container):
     
        part_two_x = [point[0] for point in self.data_collection.current_points]
        part_two_y = [point[1] for point in self.data_collection.current_points]

        # fig, axs = plt.subplots(2)
        # axs[0].imshow(np.array(self.data_collection.current_image))
        # axs[1].imshow(np.array(self.data_collection.current_image))
        # axs[0].plot(self.data_collection.x_red, self.data_collection.y_red, 'ro', color='r', markersize=4)
        # axs[0].plot(self.data_collection.x_green, self.data_collection.y_green, 'ro', color='g', markersize=4)
        # axs[1].plot(part_two_x, part_two_y, 'ro', color='b', markersize=4)
        # plt.show()

        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, self.focal, self.pp)
        norm_rot_pts = SFM.rotate(norm_prev_pts, R)
        rot_pts = SFM.unnormalize(norm_rot_pts, self.focal, self.pp)
        foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), self.focal, self.pp))
        
        fig, (tfl_attention, tfl_machine, tfl_dis) = plt.subplots(3, 1, figsize=(12,6))

        tfl_attention.imshow(np.array(self.data_collection.current_image))
        tfl_attention.plot(self.data_collection.x_red, self.data_collection.y_red, 'ro', color='r', markersize=2)
        tfl_attention.plot(self.data_collection.x_green, self.data_collection.y_green, 'ro', color='g', markersize=2)
        
        tfl_machine.imshow(np.array(self.data_collection.current_image))
        tfl_machine.plot(part_two_x, part_two_y, 'ro', color='r', markersize=2)

        #tfl_dis.set_title('curr(' + str(self.data_collection.em[0]+1) + ')')
        tfl_dis.imshow(curr_container.img)
        curr_p = curr_container.traffic_light
        tfl_dis.plot(curr_p[:,0], curr_p[:,1], 'ro', color='r', markersize=2)

        for i in range(len(curr_p)):
            tfl_dis.plot([curr_p[i,0], foe[0]], [curr_p[i,1], foe[1]], 'b')
            if curr_container.valid[i]:
                tfl_dis.text(curr_p[i,0], curr_p[i,1], r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
        tfl_dis.plot(foe[0], foe[1], 'r+')
        tfl_dis.plot(rot_pts[:,0], rot_pts[:,1], 'g+')
        plt.show()    
        

    
   

    
    





        
  