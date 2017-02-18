#!/bin/bash

F1="/Users/balder/Documents/Uni/masterarbeit/data/weka/speech/"
F2="/Users/balder/Documents/Uni/masterarbeit/data/weka/music/"
OUTPUT="/Users/balder/Documents/Uni/masterarbeit/data/weka/musicAndSpeech/"

TMP="mergAllMusicSpeech.tmp"


F1_STAR="${F1}*"
for f in $F1_STAR
do
	fnameExt="${f##*/}"
	fname="${fnameExt%.*.*}"
	
	fspeech=$f
	fmusic="${F2}${fname}.music.arff"
	
	if [ -f "$fmusic" ]
	then
		./mergArff.sh $fspeech $fmusic "${OUTPUT}${fname}.ms.arff"
	else
		echo "$fmusic not found."
	fi
done

rm -f $TMP