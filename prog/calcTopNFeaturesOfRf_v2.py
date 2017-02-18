import numpy as np
import os.path
import sys
import operator


#def printMessage(msg):
#	print "[" + os.path.basename(__file__) + "] " + msg


def run():

	if len(sys.argv) < 2:
		print "Too few parameters."
		exit(1)
	
	files = sys.argv[1:]

	#files = [ 
	#"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h1.RandomForest-out", 
	#"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h4.RandomForest-out", 
	#"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h7.RandomForest-out", 
	#"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h10.RandomForest-out", 
	#"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h13.RandomForest-out", 
	#"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h15.RandomForest-out", 
	#"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h16.RandomForest-out", 
	#"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h19.RandomForest-out", 
	#]
	
	i = 0
	
	featureLevel = {} # empty dictonary
	
	for fname in files:
		with open(fname) as f:
		
			featureLevelPerFile = {} # empty dictonary
			
			foundListing = False
			
			for line in f:
			
				# Skip empty lines
				if not line.strip():
					continue
				
				# Indicates the end of the importance listing
				if (line.startswith('Time taken to build model: ')):
					break # stop here
		
				# Indicates the start of the importance listing
				if ((not foundListing) and line.startswith('Attribute importance based on average impurity decrease (and number of nodes using that attribute)')):
					foundListing = True
					continue
				
				# In the importance listing
				if (foundListing):
					#     0.503 (   492)  AutoCorrelation11_006
					average = float(line.rpartition('(')[0].translate(None, ' '))
					occurenceInTrees = int(line.rpartition(')')[0].rpartition('(')[2].translate(None, ' '))
					feature = line.rpartition(')')[2].translate(None, ' ').translate(None, '\n')
					
					featureLevelPerFile[feature] = [ average, occurenceInTrees ]
			
		
		for feature in featureLevelPerFile:
			# Add the level to the already stored one
			if feature in featureLevel:
				summedAverage = featureLevel[feature][1] + featureLevelPerFile[feature][0]
				occurenceAverage = featureLevel[feature][2] + 1
				totalOccurenceInTrees = featureLevel[feature][3] + featureLevelPerFile[feature][1]
				featureLevel[feature] = [ (summedAverage*1.0)/occurenceAverage, summedAverage, occurenceAverage, totalOccurenceInTrees ]
			else:
				featureLevel[feature] = [ featureLevelPerFile[feature][0], featureLevelPerFile[feature][0], 1, featureLevelPerFile[feature][1] ]		
			
			
	sorted_x = reversed(sorted(featureLevel.items(), key=operator.itemgetter(1)))
	selected_x = []
	i = 1
	print "# : feature, average impurity decrease, number of nodes using that attribute"
	for f in sorted_x:
		selected_x.append([ i, f[0], f[1][0], f[1][3] ])
		i = i+1

	template = "{0:4}|{1:40}|{2:15}|{3:8}"
	print template.format("#", "FEATURE", "AVG. DECREASE", "#NODES") # header
	for rec in selected_x: 
		print template.format(*rec)

		
run()