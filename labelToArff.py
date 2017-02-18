import numpy as np
import csv
import os.path
import sys


def printMessage(msg):
	print "[" + os.path.basename(__file__) + "] " + msg

# creates an np.array which contains for each step the class-label
def readLabelFile(file, stepSize) :
	convertfunc = lambda x: int(x.strip(";"))
	timeDurations = np.genfromtxt(file, delimiter=',', usecols=(0, 1))
	labels = np.genfromtxt(file, delimiter=',', usecols=2, converters={2: convertfunc})
	
	totalTime = 0.0
	startTimes = np.empty(())

	if len(timeDurations) < 3 : # if there is only one line
		totalTime = timeDurations[0] + timeDurations[1]
		startTimes = np.array([ np.rint(timeDurations[0] / stepSize).astype(int) ])
		labels = np.array([ labels ]) #convert to array unified processing later
	else :
		totalTime = timeDurations[-1,0] + timeDurations[-1,1]
		startTimes = np.rint(timeDurations[:,0] / stepSize).astype(int)

	totalSize = int(totalTime / stepSize)
	values = np.zeros( (totalSize,), dtype=np.int )
	
	i = 0
	while i < startTimes.shape[0]-1 :
		values[startTimes[i]:startTimes[i+1]] = labels[i]
		i += 1
	values[startTimes[i]:] = labels[i]
	
	# name the labels already
	namedValues = np.where(values == 0, 'non_commercial', 'commercial')
	
	return totalSize, namedValues


def readFeatureFile(file, stepSize, size) :
	timeVals = np.genfromtxt(file, delimiter=',')	
	hasBeenSets = np.zeros( (size,), dtype=np.bool )
	values = np.zeros( (size,))
	
	for tv in timeVals :
		time = int(tv[0] / stepSize)
		if time < size :
			# if there are more per timeframe then add them!
			if hasBeenSets[time] :
				printMessage("Values are added because there are more values per timeframe.")
			values[time] += tv[1]
			hasBeenSets[time] = True
		else :
			printMessage("Featurefile (" + file + ") contains instances that are not labled. Following entry has been skipped: '" + str(tv) + "'")
	
	
	# if a timeframe has not been set, copy it from a previous timeframe
	prevVal=0.0
	for i,hasBeenSet in enumerate(hasBeenSets) :
		if hasBeenSet :
			prevVal = values[i]
		else :
			values[i] = prevVal
	
	return values


def writeArffFile(file, features, labels) :
	# write header
	with open(file, "wb") as f :
		f.write("@RELATION GENERIC\r\n")
		for i in xrange(0, features.shape[1]) :
			f.write("@ATTRIBUTE f" + '{0:06d}'.format(i+1) + " REAL\r\n")
		f.write("@ATTRIBUTE class {non_commercial,commercial}\r\n")
		f.write("@DATA\r\n")
	
	# write values
	with open(file, "ab") as f :
		writer = csv.writer(f)
		for i,label in enumerate(labels):
			row = (['{0:.16f}'.format(x) for x in features[i]])
			row.extend([label])
			writer.writerow(row)


def printUsage(message):
	print
	printMessage(message)
	print "Usage:"
	print "python " + sys.argv[0] + " <stepSize in seconds> <labelFile> <outputFile> <featureFiles> ..."
	print "python " + sys.argv[0] + " 0.2 h1.label h1.arff ff1.txt ff2.txt"
	print
	print "### labelFile example | <start>,<duration>,<class>; ###"
	print "0,3542.5,0;"
	print "3542.5,44.5,2;"
	print "3587,13,2;"
	print
	print "### featureFile example | <time>,<value> ###"
	print "0.000000000,0.0000443308199465"
	print "1.486077097,0.0000562888617424"
	print "2.972154195,0.0000505937914568"
	print


def run():
	'''
	stepSize = 0.2 #in seconds
	labelFile = "/Users/balder/Documents/Uni/masterarbeit/data/weka/ARD-h16.label"
	outputFile = "/Users/balder/Documents/Uni/masterarbeit/data/weka/ARD-h16.txt"
	featureFiles = [ "/Users/balder/Documents/Uni/masterarbeit/data/weka/ARD-h16_foo.txt", "/Users/balder/Documents/Uni/masterarbeit/data/weka/ARD-h16_foo2.txt" ]
	'''
	
	if len(sys.argv) < 4:
		printUsage("Too few parameters.")
		exit(1)
	
	stepSize = np.float(sys.argv[1])
	labelFile = sys.argv[2]
	outputFile = sys.argv[3]
	featureFiles = sys.argv[4:]
	
	# check if files exist
	if ( (not os.path.isfile(labelFile)) or os.stat(labelFile).st_size == 0 ):
		printMessage("Labelfile does not exit or is empty: " + labelFile)
		exit(1)
	for ff in featureFiles :
		if ( (not os.path.isfile(ff)) or os.stat(ff).st_size == 0 ):
			printMessage("Featurefile does not exit or is empty: " + ff)
			exit(1)
	
	# construct label array
	size, labels = readLabelFile(labelFile, stepSize)
	
	# convert the times and construct an array for each feature file
	outputFeatures = np.empty((size,0))
	for featureFile in featureFiles :
		f = readFeatureFile(featureFile, stepSize, size)
		outputFeatures = np.column_stack((outputFeatures, f))
	
	writeArffFile(outputFile, outputFeatures, labels)


run()

