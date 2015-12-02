import os
import cv2
import numpy as np
import scipy.io as sio

#TODO
#Make it a parameter as to whether we want to do sampled or full cropping of videos.
#Look at optical flow for example

def foveaCrop(fileName):
    print "Now processing: "+fileName.split("/")[5]
    cap = cv2.VideoCapture(fileName)

    #Checking the number of frames in the video
    length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    x = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    y = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

    #Get 30 equally spaced frames from the video
    sample = int(length / 30)

    if(sample==0):
        sample = 4

    #Saves the indicies of the sampled frames
    valid_frames = [sample*i for i in xrange(0,length/sample)]

    #Open a videowriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    newFileName = "../../data/pre-process/cropped_sampled/"+fileName.split('/')[4]+"/"+fileName.split('/')[5].split('.')[0]+"_cropped_sampled.avi"
    out = cv2.VideoWriter(newFileName ,fourcc, 5, (x/2,y/2))

    frame_num = 0
    while(1):
        frame_num += 1
        ret, frame = cap.read()

        #If no more frames can be read then break out of our loop
        if(not(ret)):
            break

        #Only crop every frame in the defined set
        if(frame_num in valid_frames):
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
        directory = "../../data/pre-process/cropped_sampled/"+dirs[i].split("/")[4]
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