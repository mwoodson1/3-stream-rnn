import numpy as np 
import pickle 

import logging

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, DropoutBinary, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule, Adadelta
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.data import ImgMaster
from neon.callbacks.callbacks import Callbacks
from neon.data import DataIterator

from neon.transforms import Logistic, CrossEntropyBinary, Misclassification

def get_data():
	"""
	Retrieves the stored data and reshapes it to correct data 
	iterator shape.
	"""
	#Import the randomly selected frames and labels
	X_train = np.load("training_frames.npy")
	y_train = np.load("training_frames_classes.npy")
	X_test = np.load("testing_frames.npy")
	y_test = np.load("testing_frames_classes.npy")

	print "Preparing data..."
	num_ex, x, y, rgb = X_train.shape
	#Need to reshape X and y to appropriate dataiterator formats
	X_train = np.reshape(X_train,(num_ex, x*y*rgb))

	num_ex, x, y, rgb = X_test.shape
	#Need to reshape X and y to appropriate dataiterator formats
	X_test = np.reshape(X_test,(num_ex, x*y*rgb))

	y_train = np.reshape(y_train, y_train.shape[0]).astype(np.uint8)
	y_test = np.reshape(y_test, y_test.shape[0]).astype(np.uint8)

	return (X_train, y_train, X_test, y_test)

def constuct_network():
	"""
	Constructs the layers of the AlexNet architecture.
	"""
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
	          Affine(nout=101, init=Gaussian(scale=0.01), bias=Constant(-7), activation=Softmax())]
	return Model(layers=layers)


def main():
	# parse the command line arguments
	parser = NeonArgparser(__doc__)

	args = parser.parse_args()

	logger = logging.getLogger()
	logger.setLevel(args.log_thresh)

	print "Loading data..."
	(X_train, y_train, X_test, y_test) = get_data()
	X_train = X_train[0:256,:]
	y_train = y_train[0:256]

	X_test = X_test[0:256,:]
	y_test = y_test[0:256]

	print "Training matrix dimensions"
	print "X_train: ", X_train.shape
	print "y_train: ", y_train.shape

	print "Testing matrix dimensions"
	print "X_test: ", X_test.shape
	print "y_test: ", y_test.shape

	#Create dataiterator object
	train = DataIterator(X_train, y_train, nclass=101, lshape=(3,120,160))
	test = DataIterator(X_test, y_test, nclass=101, lshape=(3,120,160))

	print "Constructing network..."
	#Create AlexNet architecture
	model = constuct_network()

	model.load_weights(args.model_file)

	# drop weights LR by 1/250**(1/3) at epochs (23, 45, 66), drop bias LR by 1/10 at epoch 45
	weight_sched = Schedule([22, 44, 65], (1/250.)**(1/3.))
	opt_gdm = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=weight_sched)
	opt_biases = GradientDescentMomentum(0.02, 0.9, schedule=Schedule([44], 0.1))
	opt = MultiOptimizer({'default': opt_gdm, 'Bias': opt_biases})

	# configure callbacks
	valmetric = TopKMisclassification(k=5)
	callbacks = Callbacks(model, train, eval_set=test, metric=valmetric, **args.callback_args)

	cost = GeneralizedCost(costfunc=CrossEntropyMulti())

	#flag = input("Press Enter if you want to begin training process.")
	print "Training network..."
	print args.epochs
	model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
	mets = model.eval(test, metric=valmetric)

	print 'Validation set metrics:'
	print 'LogLoss: %.2f, Accuracy: %.1f %%0 (Top-1), %.1f %% (Top-5)' % (mets[0], 
																		(1.0-mets[1])*100,
																		(1.0-mets[2])*100)


if __name__ == '__main__':
	main()