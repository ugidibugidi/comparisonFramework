#!/bin/bash

# -U * (Use unpruned tree.)
# -C * (Set confidence threshold for pruning. (Default: 0.25))
# -M * (Set minimum number of instances per leaf. (Default: 2))
# -R * (Use reduced error pruning. No subtree raising is performed.)
# -N * (Set number of folds for reduced error pruning. One fold is used as the pruning set. (Default: 3))
# -B (Use binary splits for nominal attributes.)
# -S * (Don't perform subtree raising.)
# -L (Do not clean up after the tree has been built.)
# -A (If set, Laplace smoothing is used for predicted probabilites.)
# -Q (The seed for reduced-error pruning.)
#
# M: 1,2,4,8,16 .... 512
# U: on/off
# C: 0.1, 0.25, 0.5
# S: on/off
# N: 2,3,4,5

cls="J48 weka.classifiers.trees.J48"

for (( m=1; m<513; m=2*m )); do
	echo "$cls -M $m -U"
	echo "$cls -M $m -C 0.1 -S"
	echo "$cls -M $m -C 0.1"
	echo "$cls -M $m -C 0.25 -S"
	echo "$cls -M $m -C 0.25"
	echo "$cls -M $m -C 0.5 -S"
	echo "$cls -M $m -C 0.5"
	for (( n=2; n<6; n++ )); do
		echo "$cls -M $m -R -N $n"
	done
done

