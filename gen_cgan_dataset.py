import random
from math import pi, sin, cos
from folding import performe_folding
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2



def main():

    folded_imgs=performe_folding("input/unfolded_real_mask.png",1000)
    #folded_imgs=performe_folding("input/unfolded.png",10)
    c_img_data=[]
    d_img_data=[]
    for fold in folded_imgs:
        c_img=fold.copy()
        c_img[c_img>0]=1
        c_img_data.append(np.expand_dims(c_img,axis=-1))
        d_img_data.append(np.expand_dims(fold,axis=-1))

    #norm the foldf 0-1
    d_img_data=list(np.array(d_img_data)/np.max(np.array(d_img_data)))

    #save the datasets
    with open("datasets/cgan_dataset.pkl", 'wb') as f:
        pickle.dump((c_img_data,d_img_data), f)

    #example output
    example_imgs_heat=[]
    for i in range(100):
        img=(d_img_data[random.randint(0,len(d_img_data)-1)]*255).astype('uint8')
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET)
        bordersize=5
        border = cv2.copyMakeBorder(
                                    img,
                                    top=bordersize,
                                    bottom=bordersize,
                                    left=bordersize,
                                    right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0]
                                    )
        example_imgs_heat.append(border)

    #plot grid
    rows=[]
    for i in range(0,100,10):
        t_row=example_imgs_heat[i:i+10]
        x=np.concatenate([t_row[y] for y in range(10)],axis=1)  
        rows.append(x)
    data_example=np.concatenate([rows[y] for y in range(len(rows))],axis=0) 

    cv2.imwrite("dataset_example_cvjet.png", data_example)


   


if __name__== "__main__":
  main()
