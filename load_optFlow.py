import os
import sys
import cv2
import numpy as np
import scipy.io as sio

def aviToImg(filename, testData, BWFLAG):
	print filename

	cap = cv2.VideoCapture(filename)
	while not cap.isOpened():
		cap = cv2.VideoCapture(new_fileName)
		cv2.waitKey(1000)
		print "Wait for the header"

	tmp = filename.split("/")
	pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	out = 0
	wait = 0
	while(1):
		ret, frame2 = cap.read()
		#If no more frames can be read then break out of our loop
		if(not(ret)):
			break
		else:
			pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
			if(testData):
				if(BWFLAG):
					file_dir = "testData/optFlow_BW/"+tmp[3]+"/"
				else:
					file_dir = "testData/optFlow/"+tmp[3]+"/"
			else:
				if(BWFLAG):
					file_dir = "trainData/optFlow_BW/"+tmp[3]+"/"
				else:
					file_dir = "trainData/optFlow/"+tmp[3]+"/"
			img_Name = tmp[4].split(".")[0] + "_" + str(int(pos_frame))
			cv2.imwrite(file_dir+img_Name+".jpg",frame2)
			out += 1

	#Close the file
	cap.release()
	return

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
	        fName = "data/UCF-101/"+line.rstrip()
	        testVids[fName] = class_ind[line.split("/")[0]]
	return testVids

def createTrainingSet(dirs,BWFLAG):
	#Get all the videos in the training list
	trainVids = getTrainList()
	keys = trainVids.keys()

	#Dictionary to hold filename and the list of random frame indices 
	trainingFrames = dict.fromkeys(keys)

	l = len(dirs)
	#Loop through all directories
	for i in xrange(l):
		print("%.2f" % (float(i)/float(l)))
		if(i==0):
			continue
		#Loop through every file in the directory
		for filename in os.listdir(dirs[i]):
			splt = filename.split('_')
			new_fileName = splt[0]+'_'+splt[1]+'_'+splt[2]+'_'+splt[3]+'.avi'
			#print 'data/UCF-101/'+dirs[i].split('/')[3]+new_fileName
			#Check if filename exists in training set, otherwise skip
			if 'data/UCF-101/'+dirs[i].split('/')[3]+'/'+new_fileName in keys:
				aviToImg(dirs[i]+"/"+filename, 0, BWFLAG)
	return

def createTestingSet(dirs,BWFLAG):
	#Get all the videos in the training list
	trainVids = getTestList()
	keys = trainVids.keys()

	l = len(dirs)
	cnt = 0
	#Loop through all directories
	for i in xrange(l):
		print("%.2f" % (float(i)/float(l)))
		if(i==0):
			continue
		#Loop through every file in the directory
		for filename in os.listdir(dirs[i]):
			new_fileName = splt[0]+'_'+splt[1]+'_'+splt[2]+'_'+splt[3]+'.avi'
			#Check if filename exists in training set, otherwise skip
			if 'data/UCF-101/'+dirs[i].split('/')[3]+'/'+new_fileName in keys:
				aviToImg(dirs[i]+"/"+filename, 1, BWFLAG)

def main(argv):

    if(argv and argv[0]=="-bw"):
        print "Loading BW optical flow"
        BWFLAG = 1
    else:
        print "Loading RGB optical flow"
        BWFLAG = 0

	#Get all the directories in the UCF-101 dataset
	if(BWFLAG==0):
		dirs = [x[0] for x in os.walk("data/pre-process/optFlow/")]
	else:
		dirs = [x[0] for x in os.walk("data/pre-process/optFlow_BW/")]
	l = len(dirs)

	for i in xrange(l):
		if(i==0):
			continue
		directory = "trainData/optFlow/"+dirs[i].split("/")[3]
		if not os.path.exists(directory):
			os.makedirs(directory)

	createTrainingSet(dirs,BWFLAG)

	for i in xrange(l):
		if(i==0):
			continue
		directory = "testData/optFlow/"+dirs[i].split("/")[3]
		if not os.path.exists(directory):
			os.makedirs(directory)

	createTestingSet(dirs,BWFLAG)

	return

if __name__ == '__main__':
	main(sys.argv[1:])