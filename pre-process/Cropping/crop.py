import os
import cv2
import numpy as np
import scipy.io as sio

def foveaCrop(fileName):
    print "Now processing: "+fileName.split("/")[5]
    cap = cv2.VideoCapture(fileName)

    #Checking the number of frames in the video
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    x = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    y = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    #Open a videowriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    newFileName = "../../data/pre-process/cropped/"+fileName.split('/')[4]+"/"+fileName.split('/')[5].split('.')[0]+"_cropped.avi"
    out = cv2.VideoWriter(newFileName ,fourcc, 20, (x/2,y/2))

    while(1):
        ret, frame = cap.read()

        #If no more frames can be read then break out of our loop
        if(not(ret)):
            break

        frame_height, frame_width, RGB = frame.shape

        height_offset = frame_height/4
        width_offset = frame_width/4

        #Take the center part of the frame
        new_frame = frame[height_offset:(y/2)+height_offset, width_offset:(x/2)+width_offset, :]

        #Output to new avi file
        out.write(new_frame)

    # Release everything if job is finished
    cap.release()
    out.release()


def main():
    #Get all the directories in the UCF-101 dataset
    dirs = [x[0] for x in os.walk("../../data/UCF-101/")]

    #Creating a directory similar to dataset for the pre-processed data
    for i in xrange(len(dirs)):
        if(i==0):
            continue
        directory = "../../data/pre-process/cropped/"+dirs[i].split("/")[4]
        if not os.path.exists(directory):
            os.makedirs(directory)

    l = len(dirs)
    #Loop through all directories
    for i in xrange(l):
        print("%.2f" % (float(i)/float(l)))
        if(i==0):
            continue
        #Loop through every file in the directory
        for filename in os.listdir(dirs[i]):
            foveaCrop(dirs[i]+"/"+filename)


if __name__ == '__main__':
    main()