import numpy as np 
import pickle 

import logging

from neon.util.argparser import NeonArgparser
from neon.initializers import Constant, Gaussian
from neon.layers import Conv, DropoutBinary, Pooling, GeneralizedCost, Affine
from neon.optimizers import GradientDescentMomentum, MultiOptimizer, Schedule, Adadelta
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, TopKMisclassification
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from neon.data import DataIterator

from neon.data import ImageLoader

from neon.transforms import Logistic, CrossEntropyBinary, Misclassification
from neon.data import ImgMaster


def constuct_network():
	"""
	Constructs the layers of the CNN architecture described in 
	http://yerevann.github.io/2015/10/11/spoken-language-identification-with-deep-convolutional-networks/.
	"""
	layers = [Conv((7, 7, 20), init=Gaussian(scale=0.01), bias=Constant(0), 
                   activation=Rectlin(),padding=3, strides=1),
	          Pooling(3, strides=2),
	          Conv((5, 5, 20), init=Gaussian(scale=0.01), bias=Constant(1), 
                   activation=Rectlin(), padding=2, strides=1),
	          Pooling(3, strides=2),
	          Conv((3, 3, 20), init=Gaussian(scale=0.03), bias=Constant(0), 
                   activation=Rectlin(), padding=1, strides=1),
              Pooling(3,strides=2),
	          Conv((3, 3, 20), init=Gaussian(scale=0.03), bias=Constant(1), 
                   activation=Rectlin(), padding=1, strides=1),
              Pooling(3,strides=2),
	          Conv((3, 3, 20), init=Gaussian(scale=0.03), bias=Constant(1), 
                   activation=Rectlin(), padding=1, strides=1),
	          Pooling(3, strides=2),
	          Conv((3, 3, 20), init=Gaussian(scale=0.03), bias=Constant(1), 
                   activation=Rectlin(), padding=1, strides=1),
              Pooling(3,strides=2),
	          Affine(nout=1024, init=Gaussian(scale=0.01), bias=Constant(1), 
                     activation=Rectlin()),
	          DropoutBinary(keep=0.5),
	          Affine(nout=4096, init=Gaussian(scale=0.01), bias=Constant(1), 
                     activation=Rectlin()),
	          DropoutBinary(keep=0.5),
	          Affine(nout=101, init=Gaussian(scale=0.01), bias=Constant(-7), 
                     activation=Softmax())]
	return Model(layers=layers)


def main():
	# parse the command line arguments
	parser = NeonArgparser(__doc__)

	args = parser.parse_args()

	logger = logging.getLogger()
	logger.setLevel(args.log_thresh)

	#Set up batch iterator for training images
	train = ImgMaster(repo_dir='spectroDataTmp', set_name='train', inner_size=400, subset_pct=100)
	val = ImgMaster(repo_dir='spectroDataTmp', set_name='validation', inner_size=400, subset_pct=100, do_transforms=False)
	test = ImgMaster(repo_dir='spectroTestDataTmp', set_name='validation', inner_size=400, subset_pct=100, do_transforms=False)

	train.init_batch_provider()
	test.init_batch_provider()

	print "Constructing network..."
	model = constuct_network()

	model.load_weights(args.model_file)

	#Optimizer
	opt = Adadelta()
 
	# configure callbacks
	valmetric = TopKMisclassification(k=5)
	callbacks = Callbacks(model, train, eval_set=val, metric=valmetric, **args.callback_args)

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
	test.exit_batch_provider()
	train.exit_batch_provider()

if __name__ == '__main__':
	main()
