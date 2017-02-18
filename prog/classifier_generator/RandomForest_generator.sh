#!/bin/bash

# -depth * The maximum depth of the trees, 0 for unlimited
# -K * The number of attributes to be used in random selection
# -I * The number of trees to be generated
# -S The random number seed to be u
#
# K: 1,2,4,8,16,32,64,128
# I: 1,2,4,8,16,32,64,128,256,512
# depth: 0(=unlimited)


cls="RandomForest weka.classifiers.trees.RandomForest"

for (( k=1; k<129; k=2*k )); do
	for (( i=1; i<513; i=2*i )); do
		#for (( d=1; d<33; d=2*d )); do
		#	echo "$cls -K $k -I $i -depth $d"
		#done
		echo "$cls -K $k -I $i -depth 0"
	done
done

