import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py

#for filename in os.listdir("../data/UCF-101/"):
dirs = [x[0] for x in os.walk("../data/UCF-101/")]

for i in xrange(len(dirs)):
    if(i==0):
        continue
    directory = "../data/pre-process/"+dirs[i].split("/")[3]
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_optFlow(fileName,Dir,vidName):
    directory = "../data/pre-process/"+Dir.split("/")[3]
    cap = cv2.VideoCapture(fileName)

    #Start reading frames
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    #Checking the number of frames in the video
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    #Store the flow images in a list for now
    flow_vid = []
    flow_vid.append(np.zeros(frame1.shape))

    #Loop through all of the frames and calculate optical flow
    while(1):
        ret, frame2 = cap.read()

        #If no more frames can be read then break out of our loop
        if(not(ret)):
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

        flow_vid.append(rgb)

        prvs = next

    #Close the file
    cap.release()

    #Convert our list of flow frames to an array
    flow_vid = np.array(flow_vid)

    #Store as a .mat file
    print directory+"/"+vidName.split(".")[0]+"_oflow"
    sio.savemat(directory+"/"+vidName.split(".")[0]+"_oflow.mat", {'flow':flow_vid})

#Loop through all directories
for i in xrange(len(dirs)):
    if(i==0):
        continue
    #Loop through every file in the directory
    for filename in os.listdir(dirs[i]):
        new_file = filename.split('.')[0]+"_oflow.mat"
        make_optFlow(dirs[i]+"/"+filename,dirs[i],filename)