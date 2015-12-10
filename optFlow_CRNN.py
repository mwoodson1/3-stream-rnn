import numpy as np 
import pickle 

import logging

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, DropoutBinary, Pooling, GeneralizedCost, Affine, LSTM
from neon.optimizers import Adadelta
from neon.transforms import Rectlin, Tanh, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.data import DataIterator, ImgMaster

def constuct_network():
	"""
	Constructs the layers of our RCNN architecture. It is similar to AlexNet but simplified to only a 
	few convolutional layers and 3 LSTM layers.
	"""
	layers = [Conv((11, 11, 64), init=Gaussian(scale=0.01), bias=Constant(0), activation=Rectlin(),
	               padding=3, strides=4),
			  Pooling(3, strides=2),
	          Conv((7, 7, 128), init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin(),
	               padding=2),
	          Pooling(3, strides=2),
	          Conv((5, 5, 256), init=Gaussian(scale=0.03), bias=Constant(0), activation=Rectlin(),
	               padding=1),
	          Conv((3, 3, 256), init=Gaussian(scale=0.03), bias=Constant(1), activation=Rectlin(),
	               padding=1),
	          Pooling(3, strides=2),
	          Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), activation=Rectlin()),
	          DropoutBinary(keep=0.5),
	          LSTM(512, init=Gaussian(scale=0.03), activation=Rectlin(), gate_activation=Tanh()),
	          LSTM(512, init=Gaussian(scale=0.03), activation=Rectlin(), gate_activation=Tanh()),
	          LSTM(512, init=Gaussian(scale=0.03), activation=Rectlin(), gate_activation=Tanh()),
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

	#Set up batch iterator for training images
	train = ImgMaster(repo_dir='dataTmp_optFlow_BW', set_name='train', inner_size=240, subset_pct=100)
	val = ImgMaster(repo_dir='dataTmp_optFlow_BW', set_name='train', inner_size=240, subset_pct=100, do_transforms=False)
	test = ImgMaster(repo_dir='dataTestTmp_optFlow_BW', set_name='train', inner_size=240, subset_pct=100, do_transforms=False)

	train.init_batch_provider()
	val.init_batch_provider()
	test.init_batch_provider()

	print "Constructing network..."
	#Create AlexNet architecture
	model = constuct_network()

	# Optimzer for model
	opt = Adadelta()

	# configure callbacks
	valmetric = TopKMisclassification(k=5)
	callbacks = Callbacks(model, train, eval_set=test, metric=valmetric, **args.callback_args)

	cost = GeneralizedCost(costfunc=CrossEntropyMulti())

	#flag = input("Press Enter if you want to begin training process.")
	print "Training network..."
	model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
	mets = model.eval(test, metric=valmetric)

	print 'Validation set metrics:'
	print 'LogLoss: %.2f, Accuracy: %.1f %%0 (Top-1), %.1f %% (Top-5)' % (mets[0], 
																		(1.0-mets[1])*100,
																		(1.0-mets[2])*100)
	return 

if __name__ == '__main__':
	main()