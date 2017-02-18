import numpy as np
import os.path
import sys
import operator


#def printMessage(msg):
#	print "[" + os.path.basename(__file__) + "] " + msg


def run():
	
	files = [ 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h1.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h4.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h7.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h10.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h13.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h15.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h16.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h19.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ARD-h22.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h2.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h5.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h8.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h11.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h12.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h17.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h20.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h20.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/RTL-h23.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/Sat1-h1.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/Sat1-h4.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/Sat1-h7.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/Sat1-h10.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/Sat1-h13.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/Sat1-h16.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/Sat1-h19.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/Sat1-h22.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h2.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h5.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h8.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h11.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h14.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h15.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h17.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h20.RandomForest-out", 
	"/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/train_output/ZDF-h23.RandomForest-out"
	]
	
	
	
	i = 0
	
	featureLevel = {} # empty dictonary
	
	for fname in files:
		with open(fname) as f:
		
			featureLevelPerTree = {} # empty dictonary
			
			for line in f:
			
				# Indicates the end of the listing of the tress
				if (line.startswith('Time taken to build model:')):
					break # stop here
			
				# Skip empty lines
				if not line.strip():
					continue
		
				# Indicates the start of a new tree
				if (line.startswith('==========')):
					featureLevelPerTree = {} # empty dictonary
					continue
				
				# Indicates the end of a tree
				if (line.startswith('Size of the tree : ')):
					for feature in featureLevelPerTree:
						# Add the level to the already stored one
						if feature in featureLevel:
							summedLevel = featureLevel[feature][1] + featureLevelPerTree[feature]
							occurence = featureLevel[feature][2] + 1
							featureLevel[feature] = [ (summedLevel*1.0)/occurence, summedLevel, occurence ]
						else:
							featureLevel[feature] = [ featureLevelPerTree[feature], featureLevelPerTree[feature], 1]
				
				
				# Only count lines that list features (and only the lesser-symbol ... they do only 
				# occur in pairs therefore only count it once)
				if ('<' in line):
					# Count only highest occurence of feature (it can also occur on a lower level in 
					# another branch of the tree)
					feature = line.rpartition('<')[0].translate(None, ' ').translate(None, '|')
					level = line.count('|')
				
					# Check if it already has been counted and if it occurend nearer to the root
					# (indicated by a smaller number)
					if (feature in featureLevelPerTree):
						if (level < featureLevelPerTree[feature]):
							featureLevelPerTree[feature] = level
					else:
						featureLevelPerTree[feature] = level
			
			
	sorted_x = sorted(featureLevel.items(), key=operator.itemgetter(1))
	i = 1
	for f in sorted_x:
		print i, ": ", f
		i = i+1

		
run()