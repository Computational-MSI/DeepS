import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from model_unet_color import opt

def randon_mask(k_list, im_size, p):
    
    k_value = random.sample(k_list, 1)[0]
    
    #start=time.time()
    # print(k_value)
    N = im_size // k_value
    
    rdn1=([0]*(round(N**2*p)))

    rdn1.extend(([1]*(round(N**2*(1-p)))))
    #end=time.time()
   # start=time.time()
    #rdn1=sorted(rdn1, key=lambda _: random.random())
    np.random.shuffle(rdn1)
   # end=time.time()
    tmp=[]
    #start=time.time()
    
    
    
    #start=time.time()
    tmp = np.asarray(rdn1).reshape(N, N)
    #end=time.time()
    #start=time.time()
    tmp = tmp.repeat(k_value, 0).repeat(k_value, 1).astype('float32')
    
    #tmp=cv2.resize(tmp,(im_size,im_size,3))
    # print(rdn)
    #end=time.time()
    
   
    #print(end-start)
    return tmp

def randon_line_mask(im_size, p):
    rdn1=([0]*(round(im_size*p)))
    rdn1.extend(([1]*(round(im_size*(1-p)))))
    np.random.shuffle(rdn1)
   
    tmp=np.array(rdn1)
    #tmp = np.array([0 if i in rdn else 1 for i in range(im_size)])
    tmp=tmp.repeat(im_size, 0).reshape((im_size,im_size))
    return tmp
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

def gen_line_mask(n, im_size, partition):
    while True:
        Ms = []
        
        for i in range(n):
            tmp3 = randon_line_mask(im_size, partition)
            

            tmp=cv2.cvtColor(tmp3.astype(np.float32),cv2.COLOR_GRAY2BGR)
            
          
            
            tmp=np.transpose(tmp, (2, 0, 1))
            # print(index)
            
            Ms.append(tmp)
        
        yield Ms
#k_value = [8,2,4]
#k_value = random.sample(k_value, 1)
#Ms_generator = gen_mask(k_value, opt.batchSize, opt.imageSize,0.5)

#k_value = [2,4,8,16]
#img_size = 256
#k_value = random.sample(k_value, 1)
#Ms_generator = gen_mask(k_value, 32, img_size)
#Ms = next(Ms_generator)
#print(np.shape(Ms))
#input_image = np.load('/media/ld/Elements SE/MTBLS176/pancreas/mz=2026.npy',encoding = "latin1")[0]
#input_image = cv2.resize(input_image,(256,256))
#inputs = [input_image * mask for mask in Ms]
#plt.imshow(inputs[0])
#plt.show()
#

