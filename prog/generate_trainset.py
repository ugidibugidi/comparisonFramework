import numpy as np
import csv
import os.path
import sys


#TODO
# - better usage output
# - error handling + messages (e.g. no split is used, to few commercials ...)


def run():
	
	#numOfLinesPerFile = 2
	#splitPercent = 0.5
	#trainFile = "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/trainset/ARD-h4.train.arff1"
	#testset = [ "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/testset/RTL-h5.test.arff", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/testset/RTL-h20.test.arff" ]
	
	#python generate_trainset.py 2 0.5 /Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/trainset/ARD-h4.train.arff1 /Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/testset/RTL-h5.test.arff /Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/testset/RTL-h20.test.arff
	
	if len(sys.argv) < 4:
		print "Too few parameters."
		exit(1)
	
	numOfLinesPerFile = int(sys.argv[1])
	splitPercent = float(sys.argv[2])
	trainFile = sys.argv[3]
	testset = sys.argv[4:]
	
	
	numOfFiles = len(testset)
	
	lastCol = np.empty((0), dtype=int)
	data = []
	for testFile in testset :
		#read, remove header, convert to int, append	
		raw_lastCol = np.genfromtxt(testFile, usecols=(-1), delimiter=',', comments='@', dtype="1|S")		
		raw_lastCol[raw_lastCol == 'n'] = 0 # non-commercial
		raw_lastCol[raw_lastCol == 'c'] = 1 # commercial
		lastCol = np.append(lastCol, raw_lastCol.astype(int))	
		
		for line in open(testFile):
			li=line.strip()
			if not li.startswith("@"):
				data.append(line)
	
	
	idx = np.transpose(np.vstack((np.arange(lastCol.shape[0]), lastCol)))
	np.random.shuffle(idx) # shuffles rows
	
	finalIdx = np.empty((0), dtype=int)
	if 0.0 <= splitPercent and splitPercent <= 1.0 :
		# perform split
		shuffledIdx = idx[idx[:,1].argsort()] # sort rows by 2nd column
		numOfNonComm = int(numOfLinesPerFile * numOfFiles * (1-splitPercent))
		numOfComm = int(numOfLinesPerFile * numOfFiles * splitPercent)
		finalIdx = shuffledIdx[:numOfNonComm,0]
		finalIdx = np.append(finalIdx, shuffledIdx[-numOfComm:,0])
		np.random.shuffle(finalIdx)
	else :
		numOfLines= int(numOfLinesPerFile * numOfFiles)
		finalIdx = idx[:numOfLines,0]
	
	# append to file (because header has already been written)
	with open(trainFile, "a") as f:
		for i in finalIdx :
			#print data[i].split(',')[-1:]
			f.write(data[i])

	
run()