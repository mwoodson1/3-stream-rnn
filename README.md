#Efficient 3-Stream RNN for Action Classification
CMU 18-794 Pattern Recognition Theory final project repo.

# Instructions
## The Neon Framework
We use [Neon](https://github.com/NervanaSystems/neon) for all of our neural net architecture setup and training. Neon allowed for easy configuration of the architectures and has excelent performance time which allowed for rapid prototyping and testing of hyperparameters. We reccomend following the reccomended route of installing Neon via [virtual enviornment](http://neon.nervanasys.com/docs/latest/user_guide.html). 

## Download UCF-101 Dataset
Obtain the UCF-101 dataset from the following link. Place and extract the data in a directory called /data/ within the top lecel of this project. All other scripts will assume the data is in that directory.

## Pre-process Data
We perform a few pre-processing steps on the data such as computing optical flow, cropping the images, and selecting random samples of frames from each video. Though this does save a lot of time in the training process it does incur a higher load on storage. 

Run **./run_pre-process.sh** to perform all pre-processing. Ensure you have enough space on disk and the UCF-101 dataset is in the correct subdirectory. 

## Pre-process Spectrogram Data
In addition to cropping the images and computing optical flow, we also train on the sound produced by a majority of the videos in the UCF-101 Dataset. We pre-process the sound by extracting the sound into .wav files and then converting them into 400x400 spectrogram images. Prior to running the spectrogram scripts, make sure you have ffmpeg installed on your machine. 

How to install ffmpeg: https://trac.ffmpeg.org/wiki/CompilationGuide. 
If you have installation issues try the following guide: http://www.faqforge.com/linux/how-to-install-ffmpeg-on-ubuntu-14-04/.

Instructions to generate spectrograms of the data:

1. Run **pre-process/Spectrogram/extract-sound.py**
2. Open MATLAB
3. Run **pre-process/Spectrogram/generateSpectrogram.m** in MATLAB
4. Run **/training_spectros.py**

**NOTE:** If you run into Java memory errors when opening Matlab add the following line to your **java.opts** file in your Matlab directory: 
`-XX:-UseGCOverheadLimit`

## Convert pre-trained AlexNet weights
For our CNN we use a pre-trained AlexNet architecture. In classic AlexNet the softmax output is of size 1000 which we need to convert to 101(total number of classes in UCF-101). Running **networks/make_new_weights.py** will do the conversion for you and save the new saved weights in **my_alexnet.py**.

## Train CNN on UCF-101 Cropped Images
Since we take 70 images per video the dataset will not all fit in memory to train on. To overcome this we take advantage of Neons batch writer functionality to process our data in batches. Instructions on how to turn our dataset into batches can be found [here](http://neon.nervanasys.com/docs/latest/datasets.html).
To run just the CNN on the pre-processed cropped data, simply run **cropped_CNN.py**. You may need to change the input to the ImgMaster call depending on where you placed your data batches.

## Train CRNN on UCF-101 Optical Flow Images
Follow the same procedure to construct the batches for the optical flow images.
To run just the CRNN on the pre-processed optical flow data, simply run **cropped_CNN.py**.

## Train CNN on UCF-101 Spectrogram Images
Follow the same procedure to construct the batches for the spectrogram images.
To run just the CNN on the pre-processed spectrogram data, simply run **spectrogram_CNN.py**.
