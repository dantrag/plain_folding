import random
from math import pi, sin, cos
from folding import perform_folding
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os



def main():
    num_examples=100
    max_fold_count=3
    min_area_folding=0.2
    xy_axis_bias=1.0
    perturb=True
    folded_imgs=perform_folding("input/small_tee_mask.png",num_examples, max_fold_count, min_area_folding, xy_axis_bias,
                                save_images=True, perturb_after_each_fold=perturb)
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

    #agregate it in the style that pix2pix wants to make it easy
    pix2pix_data=[]
    for c_img, d_img in zip(c_img_data, d_img_data):
        p2p= np.concatenate((d_img,c_img),axis=1)
        pix2pix_data.append(p2p)

    folder_name="unfolding_color_" + str(num_examples) + "_fc_" + str(max_fold_count) + "_af_" + str(min_area_folding).replace(".","p") + ("_noize" if perturb else "")

    #save in folder sructure as grayscale image
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    if not os.path.exists(folder_name + "/train"):
        os.mkdir(folder_name + "/train")
    if not os.path.exists(folder_name + "/trainA"):
        os.mkdir(folder_name + "/trainA")
    if not os.path.exists(folder_name + "/trainB"):
        os.mkdir(folder_name + "/trainB")
    if not os.path.exists(folder_name + "/val"):
        os.mkdir(folder_name + "/val")
    if not os.path.exists(folder_name + "/test"):
        os.mkdir(folder_name + "/test")
    if not os.path.exists(folder_name + "/testA"):
        os.mkdir(folder_name + "/testA")
    if not os.path.exists(folder_name + "/testB"):
        os.mkdir(folder_name + "/testB")
    #shuffel
    random.shuffle(pix2pix_data)
    random.shuffle(c_img_data)
    random.shuffle(d_img_data)

    for i, p2p in enumerate(pix2pix_data):
        c_img = c_img_data[i]
        d_img = d_img_data[i]

        if i <= len(pix2pix_data)*0.8:
            cv2.imwrite(folder_name + "/train/"+str(i)+'.jpg', p2p*255)
            cv2.imwrite(folder_name + "/trainA/"+str(i)+'.jpg', c_img*255)
            cv2.imwrite(folder_name + "/trainB/"+str(i)+'.jpg', d_img*255)
        if i > len(pix2pix_data)*0.8 and i < len(pix2pix_data)*0.9:
            cv2.imwrite(folder_name + "/val/"+str(i)+'.jpg', p2p*255)
            cv2.imwrite(folder_name + "/trainA/"+str(i)+'.jpg', c_img*255)
            cv2.imwrite(folder_name + "/trainB/"+str(i)+'.jpg', d_img*255)
        if i >= len(pix2pix_data)*0.9:
            cv2.imwrite(folder_name + "/test/"+str(i)+'.jpg', p2p*255)            
            cv2.imwrite(folder_name + "/trainA/"+str(i)+'.jpg', c_img*255)
            cv2.imwrite(folder_name + "/trainB/"+str(i)+'.jpg', d_img*255)




    #example output
    example_imgs_heat=[]
    for i in range(100):
        img=(pix2pix_data[random.randint(0,len(pix2pix_data)-1)]*255).astype('uint8')
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
