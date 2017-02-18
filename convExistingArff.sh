#!/bin/bash

#DATA="ARD-h16.speech.arff"
#DATA_COLS=52 #+3
#LABELS="ARD-h16.arff"
#LABEL_COL=1
#OUTPUT="ARD-h16-comm.speech.arff"

#./convExistingArff.sh /Users/balder/Documents/Uni/masterarbeit/data/from_KV/train/speech/ARD-h16.speech.arff 52 ARD-h16.txt 1 speech/ARD-h16.speech.arff

DATA=$1
DATA_COLS=$2
LABELS=$3
LABEL_COL=$4
OUTPUT=$5

NUMBER_OF_LINES=18000

TMP1="convExistingArff.tmp"
TMP2="convExistingArff2.tmp"

tail -n $NUMBER_OF_LINES $DATA | cut -d"," -f "-${DATA_COLS}" > $TMP1
tail -n $NUMBER_OF_LINES $LABELS | cut -d"," -f $LABEL_COL > $TMP2

header=`echo "$DATA_COLS + 3" | bc`
head -n $header $DATA \
	| sed "s|no_speech|non_commercial|" \
	| sed "s|no_music|non_commercial|" \
	| sed "s|speech|commercial|" \
	| sed "s|music|commercial|" \
	> $OUTPUT

paste -d"," $TMP1 $TMP2 >> $OUTPUT

rm -f $TMP1
rm -f $TMP2