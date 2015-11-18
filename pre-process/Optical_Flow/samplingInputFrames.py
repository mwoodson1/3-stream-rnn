import os
import cv2
import numpy as np 
import scipy.io as sio
import random

NUM_FRAMES = 100  #Number of frames to be selected for each video

#Goes through all the preprocessed .mat files for each video and selects a
#random set of frames for training set
#output a dictionary of {vidName: list of frame indices selected}
def selectRandomFrames(fileName, Dir, vidName):
	directory = "../../data/pre-process/"+Dir.split("/")[4]

	cap = cv2.VideoCapture(fileName)

	#Start reading frames
	ret, frame1 = cap.read()
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255

    #Checking the number of frames in the video
	length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    #Generate unique numbers within a range (for random indices of frames)
	randomIndexSet = []
	if NUM_FRAMES > length:
		randomIndexSet = random.sample(range(0, length), length)
	else:
		randomIndexSet = random.sample(range(0, length), NUM_FRAMES)

	return randomIndexSet

#Returns a dictionary containing all the video (.avi) file names in the training list
def getTrainList():
	trainVids = {}
	directory = "../../data/ucfTrainTestlist/"
	file1 = "trainlist01.txt"
	file2 = "trainlist02.txt"
	file3 = "trainlist03.txt"
	files = [file1, file2, file3]
	for f in files:
		fOpen = open(directory+f, 'r')
		for line in fOpen:
			fName = line.split(" ")[0].split("/")[1]
			trainVids[fName] = 1
	return trainVids

def main():
	#Get all the directories in the UCF-101 dataset
	dirs = [x[0] for x in os.walk("../../data/UCF-101/")]

	#Get all the videos in the training list
	trainVids = getTrainList()

	#Dictionary to hold filename and the list of random frame indices 
	trainingFrames = {}

	#Loop through all directories
	for i in xrange(len(dirs)):
	    if(i==0):
	        continue
	    #Loop through every file in the directory
	    for filename in os.listdir(dirs[i]):
	    	#Check if filename exists in training set, otherwise skip
	    	if filename in trainVids:
	        	frames = selectRandomFrames(dirs[i]+"/"+filename,dirs[i],filename)
	        	trainingFrames[filename] = frames
	return trainingFrames

if __name__ == '__main__':
	main()