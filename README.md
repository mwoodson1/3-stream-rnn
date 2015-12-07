# PatternRec-Project
CMU 18-794 Pattern Recognition Theory final project repo.

# Instructions
## Download UCF-101 Dataset
Obtain the UCF-101 dataset from the following link. Place and extract the data in a directory called /data/ within the top lecel of this project. All other scripts will assume the data is in that directory.

## Pre-process Data
We perform a few pre-processing steps on the data such as computing optical flow, cropping the images, and selecting random samples of frames from each video. Though this does save a lot of time in the training process it does incur a higher load on storage. 

Run the following scripts to pre-process the data:
- pre-process/Cropping/crop.py
- pre-process/make_opt_flow.py 
- pre-process/make_opt_flow.py -bw
- load_ucf.py

Or just run:
./run_pre-process.sh

## Pre-process Spectrogram Data
In addition to cropping the images and computing optical flow, we also train on the sound produced by a majority of the videos in the UCF-101 Dataset. We pre-process the sound by extracting the sound into .wav files and then converting them into 400x400 spectrogram images. Prior to running the spectrogram scripts, make sure you have ffmpeg installed on your machine. How to install ffmpeg: https://trac.ffmpeg.org/wiki/CompilationGuide. 

Instructions to generate spectrograms of the data:

1. Run **pre-process/Spectrogram/extract-sound.py**
2. Open MATLAB
3. Run **pre-process/Spectrogram/generateSpectrogram.m** in MATLAB
4. Run **/training_spectros.py**

## Convert pre-trained AlexNet weights
For our CNN we use a pre-trained AlexNet architecture. In classic AlexNet the softmax output is of size 1000 which we need to convert to 101(total number of classes in UCF-101). Running **networks/make_new_weights.py** will do the conversion for you and save the new saved weights in **my_alexnet.py**.

## Train CNN on UCF-101 Cropped Images
To run just the CNN on the pre-processed cropped data, simply run **cropped_CNN.py**.

## Train CRNN on UCF-101 Optical Flow Images
To run just the CRNN on the pre-processed optical flow data, simply run **cropped_CNN.py**.

## Train CNN on UCF-101 Spectrogram Images
To run just the CNN on the pre-processed spectrogram data, simply run **spectrogram_CNN.py**.
