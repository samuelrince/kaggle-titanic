# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:20:29 2018

@author: alexandra
"""

from numpy import *

def euclideanDistance(vectorA, vectorB):
	# Computes the euclidean distance between two vectors. The
	# two vectors must have the same size.

	mA = len(vectorA) # length of vectorA
	mB = len(vectorB) # length of vectorB
	
	assert mA == mB, 'The two vectors must have the same size'
	
	distance = 0
	
	for i in range(mA):
		distance = distance + pow((vectorA[i]-vectorB[i]),2)
		
	distance = sqrt(distance)
	
	return distance


def kNN(k,traindata,trainlabel,testdata): # return the label (0 or 1) of the test data (testdata = one vector)
    
    nbtraindata,nbfeatures = traindata.shape;
    
    distances = zeros(nbtraindata);
    for i in range(nbtraindata):
        distances[i] = euclideanDistance(traindata[i,:],testdata)
    
    # Sort distances and re-arrange labels based on the distance of the instances
    idx = distances.argsort()
    trainlabel = trainlabel[idx];
    
    countlabels = zeros(2);
    
    for i in range(k):
        countlabels[trainlabel[i]];
    
    if countlabels[0]>countlabels[1]:
        return(0)
    else : 
        return(1)

def kNNgroup(k,traindata,trainlabel,testdatas):
    nbdata = testdatas.shape[0]
    result = zeros(nbdata)
    for i in range(nbdata):
        result[i] = kNN(k,traindata,trainlabel,testdatas[i])
        
    return(result)
        
    