import os
from load_ucf import getTrainList
from shutil import copyfile

def generateTrainingSpectrograms():
	"""
	Stores all the spectrogram images for training set
	"""
	#Get all the videos in the training list
	trainVids = getTrainList()

	directory = "trainData/spectrogram/"
	
	#Loop through all videos in training set and get spectrograms
	for vidName in trainVids:
		vidName = vidName.split("data/UCF-101/")[1]
		filename = vidName.split(".")[0] + ".jpg"
		source = "data/pre-process/Spectrogram/" + filename
		filename = directory + filename

		if not os.path.exists(os.path.dirname(filename)):
		    os.makedirs(os.path.dirname(filename))

		#Check if the spectrogram exists (i.e. if video has audio)
		if not os.path.isfile(source):
			#Copy the average spectrogram instead
			source = "data/pre-process/Spectrogram/" + vidName.split("/")[0] + "/average_spectro.jpg"

		#Copy the spectrogram image into training data directory
		copyfile(source, filename)

def main():
	generateTrainingSpectrograms()

if __name__ == '__main__':
	main()