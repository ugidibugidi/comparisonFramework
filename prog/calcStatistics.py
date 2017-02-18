import numpy as np
import os.path
import sys


def printMessage(msg):
	print "[" + os.path.basename(__file__) + "] " + msg


def run():

	#statisticName = "RandomForest"
	#files = [ "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/ARD-h1.RandomForest-pred", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/ARD-h16.RandomForest-pred", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/RTL-h2.RandomForest-pred", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/RTL-h20.RandomForest-pred", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/Sat1-h1.RandomForest-pred", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/Sat1-h7.RandomForest-pred", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/ZDF-h2.RandomForest-pred", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/ZDF-h17.RandomForest-pred" ]
	
	
	if len(sys.argv) < 3:
		print "Too few parameters."
		exit(1)
	
	statisticName = sys.argv[1]
	files = sys.argv[2:]
	
	print "========================================================"
	print "	" + statisticName
	print "Filename	TP	FP	TN	FN	Accuracy"
	print "--------------------------------------------------------"
	
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	
	for f in files :
		if ( (not os.path.isfile(f)) or os.stat(f).st_size == 0 ):
			printMessage("File does not exit or is empty: " + f)
		else :
			predictions = np.genfromtxt(f, delimiter=' ', usecols=(0,1), dtype=np.int)
		
			fTP = sum(1 for p in predictions if p[0] == 1 and p[1] == 1)
			fFP = sum(1 for p in predictions if p[0] == 1 and p[1] == 2)
			fTN = sum(1 for p in predictions if p[0] == 2 and p[1] == 2)
			fFN = sum(1 for p in predictions if p[0] == 2 and p[1] == 1)
		
			fname = (os.path.basename(f)).split(".")[0]
			printStats(fname, fTP, fFP, fTN, fFN)
		
			TP += fTP
			FP += fFP
			TN += fTN
			FN += fFN
		
	print "--------------------------------------------------------"
	printStats("OVERALL", TP, FP, TN, FN)
	print "========================================================"
		

def printStats(name, TP, FP, TN, FN):

	ACC = 0.0
	if (TP + FP + TN + FN) > 0 :
		ACC = (TP + TN)*1.0 / (TP + FP + TN + FN)*1.0 * 100
		
	print name + "		" + \
		str(TP)  + "	" + \
		str(FP)  + "	" + \
		str(TN)  + "	" + \
		str(FN)  + "	" + \
		'{0:.4f}'.format(ACC) 

		
run()