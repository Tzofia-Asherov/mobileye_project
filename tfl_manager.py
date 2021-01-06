import numpy as np
import matplotlib.pyplot as plt
import Filter
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
        self.loaded_model = load_model(model_path)
        self.prev_points = None
        self.prev_image = None

        

    def run(self, img, em):
        curr_container = None
        prev_container = None

        x_red, y_red, x_green, y_green = self.attention(img)
        list_predict = self.prediction(x_red, y_red, x_green, y_green, img)
        
        if em:
            curr_container, prev_container = self.calc_distances(list_predict, em, img)

        self.prev_update(list_predict, img)
        self.visualization(x_red, y_red, x_green, y_green, curr_container, prev_container, img, em, list_predict)


    def attention(self, img):
        #x_red, y_red, x_green, y_green  = att.find_tfl_lights(np.array(img),  some_threshold=1500)  
        x_red, y_red, x_green, y_green  = Filter.covulotion_images([img]) 
        
        return x_red, y_red, x_green, y_green

    def prediction(self, x_red, y_red, x_green, y_green, img):
        red_list = [(x_red[i], y_red[i]) for i in range(len(x_red))]
        green_list =   [(x_green[i], y_green[i]) for i in range(len(x_green))]


        predict_red_points = self.get_prediction__points(red_list, img) 
        predict_green_points = self.get_prediction__points(green_list, img)
        list_predict = predict_red_points + predict_green_points 
        return list_predict
 
        #self.data_collection.init_part_two(predict_red_points )
    
    
    def get_prediction__points(self, all_points, img):

        tfl_points_machine = []
        val_images = [np.array(crop(img ,point)) for point in all_points]
        

        crop_shape=(81,81)
        for i in range(len(val_images)):
            
            if val_images[i].shape == (81,81,3):
                img =val_images[i].reshape([-1]+list(crop_shape) +[3])
            
                predictions = self.loaded_model.predict(img)
                predicted_label = np.argmax(predictions, axis=-1)

                if predicted_label[0]:
                    tfl_points_machine.append((all_points[i][0], all_points[i][1]))
            
        return tfl_points_machine


    def calc_distances(self, list_predict, em, img):
        prev_container = containers.FrameContainer(self.prev_image)
        curr_container = containers.FrameContainer(img)
        curr_container.EM = em[1]
        prev_container.traffic_light = np.array(self.prev_points)
        curr_container.traffic_light = np.array(list_predict)
        curr_container = SFM.calc_TFL_dist(prev_container, curr_container, self.focal, self.pp)
        return  curr_container, prev_container 

    def prev_update(self,list_predict, img):
        self.prev_points = list_predict
        self.prev_image = img


    def visualization(self, x_red, y_red, x_green, y_green, curr_container, prev_container, img, em, list_predict):

        part_two_x = [point[0] for point in list_predict]
        part_two_y = [point[1] for point in list_predict]

     
        if em:
            norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, self.focal, self.pp)
            norm_rot_pts = SFM.rotate(norm_prev_pts, R)
            rot_pts = SFM.unnormalize(norm_rot_pts, self.focal, self.pp)
            foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), self.focal, self.pp))
            
        fig, (tfl_attention, tfl_machine, tfl_dis) = plt.subplots(3, 1, figsize=(12,6))

        tfl_attention.imshow(np.array(img))
        tfl_attention.plot(x_red, y_red, 'ro', color='r', markersize=2)
        tfl_attention.plot(x_green, y_green, 'ro', color='g', markersize=2)
        
        tfl_machine.imshow(np.array(img))
        tfl_machine.plot(part_two_x, part_two_y, 'ro', color='r', markersize=2)

        if em:
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
        

    
   

    
    





        
  