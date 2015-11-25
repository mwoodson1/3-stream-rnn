import os
import cv2
import numpy as np 
import pickle

def main():
	if(not(os.path.isfile("alexnet.p")))
	    print "Make sure to download the alexnet weights before running this script"
	    return 0

	weights = pickle.load( open( "alexnet.p", "rb" ) )

	tmp = weights['layer_params_states']

	#Change the last layer of the AlexNet architecture weights to
	#be of size 101 instead of 1000
	for i in xrange(0,16):
	    if(i==14):
	        W = tmp[i]['params']['W']
	        tmp[i]['params']['W'] = W[0:101,:]
	    if(i==15):
	        W = tmp[i]['params']['W']
	        tmp[i]['params']['W'] = W[0:101]
	        
	pickle.dump( weights, open( "my_alexnet.p", "wb" ) )

if __name__ == '__main__':
	main()