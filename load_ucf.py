import os
import cv2
import numpy as np 
import scipy.io as sio
import random
import pickle

NUM_FRAMES = 1  #Number of frames to be selected for each video

def selectRandomFrames(fileName):
	"""
	Goes through all the preprocessed .mat files for each video and selects a
	random set of frames for training set
	output a dictionary of {vidName: list of frame indices selected}
	"""
	cap = cv2.VideoCapture(fileName)

	#Start reading frames
	ret, frame1 = cap.read()

    #Checking the number of frames in the video
	length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    #Generate unique numbers within a range (for random indices of frames)
	randomIndexSet = []
	if NUM_FRAMES > length:
		randomIndexSet = random.sample(range(0, length), length)
	else:
		randomIndexSet = random.sample(range(0, length), NUM_FRAMES)
	cap.release()

	indices = sorted(randomIndexSet)

	#Loop through all of the frames and get training frames
	output_frames = []

	tmp = fileName.split("/")
	new_fileName = "data/pre-process/cropped/"+tmp[2]+"/"+tmp[3].split(".")[0] +"_cropped.avi"

	cap = cv2.VideoCapture(new_fileName)
	while not cap.isOpened():
		cap = cv2.VideoCapture(new_fileName)
		cv2.waitKey(1000)
		print "Wait for the header"

	pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	while(1):
		ret, frame2 = cap.read()
		#If no more frames can be read then break out of our loop
		if(not(ret)):
			if(len(output_frames) >= len(indices)):
				break
			else:
				output_frames.append(frame2)
				cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame+1)
				cv2.waitKey(1000)
		else:
			pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
			#print pos_frame
			if(len(output_frames) >= len(indices)):
				break
			#print "im here"
			if(pos_frame in indices):
				output_frames.append(frame2)

	#Close the file
	cap.release()

	return np.array(output_frames)

#Returns a dictionary containing all the video (.avi) file names in the training list
def getTrainList():
	"""
	Creates a dictionary of video_name -> label
	"""
	trainVids = {}
	directory = "data/ucfTrainTestlist/"
	files = ["trainlist01.txt"]
	for f in files:
	    fOpen = open(directory+f, 'r')
	    for line in fOpen:
	        fName = "data/UCF-101/"+line.split(" ")[0]
	        trainVids[fName] = int(line.split(" ")[1])
	return trainVids


def createTrainingSet():
	"""
	Creates and saves the training set of frames and labels
	"""
	#Get all the directories in the UCF-101 dataset
	dirs = [x[0] for x in os.walk("data/UCF-101/")]
	#Get all the videos in the training list
	trainVids = getTrainList()
	keys = trainVids.keys()

	#Dictionary to hold filename and the list of random frame indices 
	trainingFrames = dict.fromkeys(keys)

	frames_so_far = np.zeros([len(keys)*NUM_FRAMES,120,160,3])
	
	outputs = np.zeros([len(keys),1])

	l = len(dirs)
	cnt = 0
	#Loop through all directories
	for i in xrange(l):
		print("%.2f" % (float(i)/float(l)))
		if(i==0):
			continue
		#Loop through every file in the directory
		for filename in os.listdir(dirs[i]):
			#Check if filename exists in training set, otherwise skip
			if dirs[i]+"/"+filename in keys:
				frames_so_far[cnt,:,:,:] = selectRandomFrames(dirs[i]+"/"+filename)
				outputs[cnt] = trainVids[dirs[i]+"/"+filename] 
				cnt += 1
	np.save("training_frames",frames_so_far)
	np.save("training_frames_classes", outputs)

def getTestList():
	"""
	Creates a dictionary of video_name->label
	"""
	directory = "data/ucfTrainTestlist/"
	#Get class index dictionary
	class_ind = {}
	fOpen = open(directory+"classInd.txt", 'r')
	for line in fOpen:
		val, index = line.split(" ")
		index = index[:len(index)-2]
		class_ind[index] = int(val)

	testVids = {}
	files = ["testlist01.txt"]
	for f in files:
	    fOpen = open(directory+f, 'r')
	    for line in fOpen:
	        fName = "data/UCF-101/"+line
	        testVids[fName] = class_ind[line.split("/")[0]]
	return testVids

def createTestingSet():
	"""
	Creates and saves the testing set of frames and labels
	"""
	#Get all the directories in the UCF-101 dataset
	dirs = [x[0] for x in os.walk("data/UCF-101/")]
	#Get all the videos in the training list
	trainVids = getTestList()
	keys = trainVids.keys()

	#Dictionary to hold filename and the list of random frame indices 
	trainingFrames = dict.fromkeys(keys)

	frames_so_far = np.zeros([len(keys)*NUM_FRAMES,120,160,3])
	
	outputs = np.zeros([len(keys),1])

	l = len(dirs)
	cnt = 0
	#Loop through all directories
	for i in xrange(l):
		print("%.2f" % (float(i)/float(l)))
		if(i==0):
			continue
		#Loop through every file in the directory
		for filename in os.listdir(dirs[i]):
			#Check if filename exists in training set, otherwise skip
			if dirs[i]+"/"+filename in keys:
				frames_so_far[cnt,:,:,:] = selectRandomFrames(dirs[i]+"/"+filename)
				outputs[cnt] = trainVids[dirs[i]+"/"+filename] 
				cnt += 1
	np.save("testing_frames",frames_so_far)
	np.save("testing_frames_classes", outputs)


def main():
	if(os.path.isfile("training_frames_classes.npy")):
		print "Found training frame file"
	else:
		createTrainingSet()

	if(os.path.isfile("testing_frames_classes.npy")):
		print "Found testing frame file"
	else:
		createTestingSet()

if __name__ == '__main__':
	main()