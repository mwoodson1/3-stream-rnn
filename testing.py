import os
#import cv2
import numpy as np 
#import scipy.io as sio
import random
import pickle 

import logging

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, DropoutBinary, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImgMaster
from neon.callbacks.callbacks import Callbacks
from neon.data import DataIterator

from neon.callbacks.callbacks import Callbacks
from neon.data import DataIterator, load_mnist
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, Misclassification
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)

args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(args.log_thresh)

#Import the randomly selected frames and labels
X_train = np.load("training_frames.npy")
y_train = np.load("training_frames_classes.npy")

num_ex, x, y, rgb = X_train.shape
#Need to reshape X and y to appropriate dataiterator formats
new_X = np.reshape(X_train,(num_ex, x*y*rgb))

new_y = np.zeros((1,101,len(y_train)))

for i in xrange(len(y_train)):
	new_vec = np.zeros((1,101))
	new_vec[0,int(y_train[i,0])-1] = 1
	new_y[:,:,i] = new_vec

#Create dataiterator object
train = DataIterator(new_X, new_y, nclass=101, lshape=(120,160,3))

#Create AlexNet architecture
layers = [Conv((11, 11, 64), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(),
               padding=3, strides=4),
          Pooling(3, strides=2),
          Conv((5, 5, 192), init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin(),
               padding=2),
          Pooling(3, strides=2),
          Conv((3, 3, 384), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(),
               padding=1),
          Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
               padding=1),
          Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
               padding=1),
          Pooling(3, strides=2),
          Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
          DropoutBinary(keep=0.5),
          Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
          DropoutBinary(keep=0.5),
          Affine(nout=1000, init=Gaussian(scale=0.01), bias=Constant(-7), activation=Softmax())]
model = Model(layers=layers)

# drop weights LR by 1/250**(1/3) at epochs (23, 45, 66), drop bias LR by 1/10 at epoch 45
weight_sched = Schedule([22, 44, 65], (1/250.)**(1/3.))
opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)
opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=Schedule([44], 0.1))
opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

# configure callbacks
valmetric = TopKMisclassification(k=5)
#callbacks = Callbacks(model, train, eval_set=test, metric=valmetric, **args.callback_args)

cost = GeneralizedCost(costfunc=CrossEntropyMulti())
#flag = input("Press Enter if you want to begin training process.")
model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost)# callbacks=callbacks)