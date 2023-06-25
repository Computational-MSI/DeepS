from torch.utils import data
import os
import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mask_gen_color import gen_mask
#from scipy import misc
#import scipy
import imageio
from model_unet_color import opt
image_size = opt.imageSize


def normalization(datingDatamat):
    max_index = np.unravel_index(np.argmax(datingDatamat, axis=None), datingDatamat.shape)
    max_arr=datingDatamat[max_index]
    min_index =  np.unravel_index(np.argmin(datingDatamat, axis=None), datingDatamat.shape)
    min_arr=datingDatamat[min_index]
    ranges = max_arr - min_arr+0.0001
    norDataSet = np.zeros(datingDatamat.shape)
    #m = datingDatamat.shape[0]
    norDataSet = datingDatamat - min_arr
    norDataSet = norDataSet/ranges
    return norDataSet

class TextileData(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        #print(img_path)

        data=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        
        data = cv2.resize(data, (image_size,image_size), interpolation=cv2.INTER_LANCZOS4)
        data=cv2.cvtColor(data,cv2.COLOR_GRAY2BGR)
        data=np.transpose(data, (2, 0, 1))
        #data = np.expand_dims(data,axis=0)
        #print(data.shape)
        return normalization(data)

    def __len__(self):
        return len(self.imgs)





