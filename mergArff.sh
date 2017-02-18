#!/bin/bash

#./mergArff.sh speech/ARD-h4.speech.arff music/ARD-h4.music.arff musicAndSpeech/ARD-h4.ms.arff

F1=$1
F2=$2
OUTPUT=$3

#F1="/Users/balder/Documents/Uni/masterarbeit/data/weka/speech/ARD-h4.speech.arff"
#F2="/Users/balder/Documents/Uni/masterarbeit/data/weka/music/ARD-h4.music.arff"
#OUTPUT="/Users/balder/Documents/Uni/masterarbeit/data/weka/musicAndSpeech/ARD-h4.ms.arff"

TMP1="mergArff.tmp"
TMP2="mergArff2.tmp"

# remove header and last column (contains class labels) in the first file
grep -v '^@' $F1 | rev | cut -d"," -f 2- | rev > $TMP1
grep -v '^@' $F2 > $TMP2

# get total number of columns to create new header (-1 for class-column)
F1_NUM_COLS=`head -1 $TMP1 | tr ',' '\n' | wc -l`
F2_NUM_COLS=`head -1 $TMP2 | tr ',' '\n' | wc -l`
NUM_COLS=`echo "$F1_NUM_COLS + $F2_NUM_COLS - 1" | bc`

# print header
echo "@RELATION GENERIC" > $OUTPUT
for (( i=0; i<NUM_COLS; i++))
do
	printf "@ATTRIBUTE f%06d REAL\n" $((i+1)) >> $OUTPUT
done
echo "@ATTRIBUTE class {non_commercial,commercial}" >> $OUTPUT
echo "@DATA" >> $OUTPUT

# print concatenated data
paste -d"," $TMP1 $TMP2 >> $OUTPUT

rm -f $TMP1
rm -f $TMP2
