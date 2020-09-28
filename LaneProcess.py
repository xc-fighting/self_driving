# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
dirs = os.listdir("test_images/")
for file in dirs:
    print("the file name is:"+file)
    #read the image
    image = mpimg.imread("test_images/"+file)
    #transfer that to gray picture
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #blur the gray picture
    blur_gray = cv2.GaussianBlur(gray,(5,5),0)
    #detect the edges get a new edges picture
    edges = cv2.Canny(gray,15,150)
    #mask map with all 0
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(500, 305), (500, 305), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    #get the area of interest
    masked_edges = cv2.bitwise_and(edges,mask)
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 20
    max_line_gap = 80
    #make a buffer with same size with original picture,all black
    line_image = np.copy(image)*0
    #get the line set with masked_edges(area of interested)
    lines = cv2.HoughLinesP(masked_edges,rho,theta,threshold,np.array([]),
                           min_line_length,max_line_gap)
    #for every line in line set draw lines
    candidateX = 1000
    candidateY = -1
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 < candidateX:
                candidateX=x1
                candidateY=y1
            elif x2 < candidateX:
                candidateX=x2
                candidateY=y2
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    
    print(str(candidateX)+" "+str(candidateY))
    color_edges = np.dstack((edges,edges,edges))
    finalImg = cv2.addWeighted(image,1,line_image,0.8,0.0)
    cv2.imwrite("test_images_result/"+file,finalImg)