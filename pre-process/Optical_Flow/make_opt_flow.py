import os
import sys
import cv2
import numpy as np
import scipy.io as sio
import threading

#Creates and saves a .mat file that has all optical flow images for input video
def make_optFlow(fileName,BW):
    print "Now processing: "+fileName.split("/")[5]
    cap = cv2.VideoCapture(fileName)

    #Start reading frames
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    #Checking the number of frames in the video
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    x = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    y = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    tmp = int(length / 30)

    if(tmp==0):
        tmp = 4
    valid_frames = [tmp*i for i in xrange(0,length/tmp)]

    #Open a videowriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    if(not(BW)):
        newFileName = "../../data/pre-process/optFlow_BW/"+fileName.split('/')[4]+"/"+fileName.split('/')[5].split('.')[0]+"_optFlow.avi"
        out = cv2.VideoWriter(newFileName ,fourcc, 20, (x,y))
    else:
        newFileName = "../../data/pre-process/optFlow_BW/"+fileName.split('/')[4]+"/"+fileName.split('/')[5].split('.')[0]+"_BW_optFlow.avi"
        out = cv2.VideoWriter(newFileName ,fourcc, 6, (x,y),0)

    frame_num = 0
    #Loop through all of the frames and calculate optical flow
    while(1):
        frame_num += 1
        ret, frame2 = cap.read()

        #If no more frames can be read then break out of our loop
        if(not(ret)):
            break
        if(frame_num in valid_frames):
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

            if(not(BW)):
                out.write(rgb)
            else:
                bw = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                out.write(bw)

            prvs = next

    #Close the file
    cap.release()
    out.release()

def main(argv):

    if(argv and argv[0]=="-bw"):
        print "Doing BW optical flow"
        BWFLAG = 1
    else:
        print "Doing RGB optical flow"
        BWFLAG = 0

    #Get all the directories in the UCF-101 dataset
    dirs = [x[0] for x in os.walk("../../data/UCF-101/")]
    l = len(dirs)

    if(not(BWFLAG)):
        #Creating a directory similar to dataset for the pre-processed data
        for i in xrange(l):
            if(i==0):
                continue
            directory = "../../data/pre-process/optFlow/"+dirs[i].split("/")[4]
            if not os.path.exists(directory):
                os.makedirs(directory)
    else:
        #Creating a directory similar to dataset for the pre-processed data
        for i in xrange(l):
            if(i==0):
                continue
            directory = "../../data/pre-process/optFlow_BW/"+dirs[i].split("/")[4]
            if not os.path.exists(directory):
                os.makedirs(directory)


    
    #Loop through all directories
    for i in xrange(l):
        print("%.2f" % (float(i)/float(l)))
        if(i==0):
            continue
        #Loop through every file in the directory
        for filename in os.listdir(dirs[i]):
            make_optFlow(dirs[i]+"/"+filename, BWFLAG)

if __name__ == '__main__':
	main(sys.argv[1:])