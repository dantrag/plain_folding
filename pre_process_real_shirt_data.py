import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pickle



def mask_shirt(img_org, dim):
    hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
    # Threshold of white in HSV space
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([255, 230, 255])
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    result = cv2.bitwise_and(img_org, img_org, mask = mask)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    #find contour
    ret,thresh = cv2.threshold(gray,50,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    #find the biggest area
    cnt = max(contours, key = cv2.contourArea)

    mask_img = np.zeros(img_org.shape, np.uint8)
    cv2.drawContours(mask_img, [cnt], 0, (255,255,255), thickness=cv2.FILLED)

    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    #masked_img = cv2.bitwise_and(img_org, img_org, mask=mask_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    res = cv2.morphologyEx(mask_img,cv2.MORPH_OPEN,kernel)
    mask_img=cv2.resize(mask_img, dim, interpolation = cv2.INTER_AREA)

    mask_img[mask_img>0]=10



    return mask_img


def main():
    #read in dataset
    dataset_path="datasets/shirt_dataset_20191217_1050_20200130_1612"
    dim=(256,256)
    file = open(dataset_path+ '.pkl','rb')
    dataset = pickle.load(file)
    file.close()

    unfolded_real=dataset[0][0]
    unfolded_real_mask=mask_shirt(unfolded_real, dim)

    cv2.imwrite("input/unfolded_real.png", unfolded_real)
    cv2.imwrite("input/unfolded_real_mask.png", unfolded_real_mask)

    #make test dataset
    test_data=[]
    for entry in dataset:
        img1=entry[0]
        img2=entry[1]
        test_data.append((cv2.resize(img1, dim, interpolation = cv2.INTER_AREA),mask_shirt(img1, dim)))
        test_data.append((cv2.resize(img2, dim, interpolation = cv2.INTER_AREA),mask_shirt(img2, dim)))

    with open("datasets/cgan_dataset_real_test.pkl", 'wb') as f:
        pickle.dump(test_data, f)




if __name__== "__main__":
  main()

