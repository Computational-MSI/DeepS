import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from model_unet_color import opt


def gen_mask(k_list, n, im_size, partition):
    while True:
        Ms = []
        
        for i in range(n):
            
            tmp3 = randon_mask(k_list, im_size, partition)
            #tmp=np.zeros((3,im_size,im_size))
            #print(tmp3)
            tmp=cv2.cvtColor(tmp3.astype(np.float32),cv2.COLOR_GRAY2BGR)
            tmp=np.transpose(tmp, (2, 0, 1))
            Ms.append(tmp)
        
        yield Ms


