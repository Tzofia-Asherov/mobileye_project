import tfl_manager as tfl
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Controller:
    def __init__(self, play_list_path):

        with open(play_list_path, 'r') as play_list:
            data_play_list = play_list.readlines()
            pkl_path = data_play_list[0].replace('\n','')
            model_path =  data_play_list[1].replace('\n','')
            index_from = int(data_play_list[2].replace('\n',''))
            index_to = int(data_play_list[3].replace('\n',''))


            with open(pkl_path, 'rb') as pklfile:
                data_pkl = pickle.load(pklfile, encoding='latin1')
            focal = data_pkl['flx']
            pp = data_pkl['principle_point']

            EM_list = []
            EM_list.append(None)
            for i in range(index_from, index_to):
                EM = np.eye(4)
                EM = np.dot(data_pkl['egomotion_' + str(i) + '-' + str(i + 1)], EM)
                EM_list.append((i,EM))
            self.em_list = EM_list
          
            img_path = []
            for i in range(4, len(data_play_list)):
                img_path.append(data_play_list[i].replace('\n',''))

            self.img_list_path = img_path

            self.tfl_manager = tfl.TFL_manager(focal, pp, model_path)
            
           
            # print('pp', pp)
            # print('focal', focal )
            # print('em', self.em_list )
            # print('tflimg', self.img_list_path )

            
    def run(self):
        for i in range(len(self.img_list_path)):
    
            em = self.em_list[i]
            img =  Image.open(self.img_list_path[i])
            self.tfl_manager.run(img, em)
            
    
            
c = Controller('playlist.txt')
c.run()

