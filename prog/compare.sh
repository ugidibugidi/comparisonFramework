#!/bin/bash

# feature files in the featureFiles-directory must be:
# 1. grouped in subdirectories according to the feature
# 2. name after the hour and the feature
# e.g. featureFiles/spectral_flux_3/ARD-h1.spectral_flux_3

# do not forget "/"-character at the end
PRINT_DEBUG=true
WORKING_DIR="/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/"
STEP_SIZE=0.2
NUM_OF_INSTANCES_PER_FILE=1000 # 60*60/0.2 = 18000 #2000 for finalblend3 #1500 for finalblend3_own_station #1000 for finalblend3_all_classifiers
USE_OWN_STATION_TRAINSET=0 #0 for finalblend3 #1 for finalblend3_own_station
TRAINSET_PERCENT_SPLIT=0.5 # use negative float for no percentage split
SMOOTHING_WINDOW_LENGTHS=('0' '128' '256' '512' '1024' '2048' '4096' '8192') # 0 ... no smoothing
SMOOTHING_ITERATIONS=('1' '2' '4' '8')
SMOOTHING_VARIABLE_WINDOW='true'


# should not be modified (unless you know why)
AUDIO_LEN_SEC=3600.0 # length of a hour in seconds
AUDIO_DIR="${WORKING_DIR}audio/"
FEATURE_FILES_DIR="${WORKING_DIR}featureFiles/"
PRECOMPUTED_FEATURE_FILES_DIR="${WORKING_DIR}_backup/featureFiles/"
LABELS_DIR="${WORKING_DIR}labels/"
LABELS_PRED_DIR="${WORKING_DIR}labels_pred/"
PROG_DIR="${WORKING_DIR}prog/"
PROG_CONF_DIR="${PROG_DIR}conf/"
TESTSET_DIR="${WORKING_DIR}testset/"
TRAINSET_DIR="${WORKING_DIR}trainset/"
TMP_DIR="${WORKING_DIR}tmp/"
WEKA_MODEL_DIR="${WORKING_DIR}weka/model/"
WEKA_TRAIN_OUTPUT_DIR="${WORKING_DIR}weka/train_output/"
WEKA_PRED_DIR="${WORKING_DIR}weka/predictions/"
RUN_OUTPUT_DIR="${WORKING_DIR}_runs/generated/"
AUDIO_FILES_PLAN_CONF="${PROG_CONF_DIR}audioFilesPlan.conf"
FEATURE_PLAN_CONF="${PROG_CONF_DIR}featurePlan.conf"
CLASSIFIER_PLAN_CONF="${PROG_CONF_DIR}classifierPlan.conf"

PYTHON_PROG="/usr/bin/python"
LABEL_TO_ARFF_PROG="${PROG_DIR}labelToArff_v2.py"
GENERATE_TRAINSET_PROG="${PROG_DIR}generate_trainset.py"
#JAVA_PROG="/usr/bin/java" # old java version (1.6) which is still referenced by the system
JAVA_PROG=/Library/Internet\ Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/bin/java
#WEKA_PROG="/Applications/weka-3-6-9/weka.jar"
WEKA_PROG="/Applications/weka-3-8-1/weka.jar"
CALC_STATISTICS_PROG="${PROG_DIR}calcStatistics_v2.py"
RF_TOP_N_FEATURES_PROG="${PROG_DIR}calcTopNFeaturesOfRf_v2.py"
YAAFE_PROG_DIR="/Applications/yaffee/build"
YAAFE_PROG="${YAAFE_PROG_DIR}/bin/yaafe.py"
PRINT_EVENLY_SPACED_NUMBERS_PROG="${PROG_DIR}print_evenly_spaced_numbers.py"
PWD_PROG="/bin/pwd"

TIME_LABEL_INDICATOR="99999"

export CLASSPATH=$WEKA_PROG:$CLASSPATH
export YAAFE_PATH=$YAAFE_PROG_DIR/yaafe_extensions
export PATH=$PATH:$YAAFE_PROG_DIR/bin
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$YAAFE_PROG_DIR/lib
export PYTHONPATH=$PYTHONPATH:$YAAFE_PROG_DIR/python_packages


main() {

	# read audio files plan (hours) that should be used
	HOURS=`cat $AUDIO_FILES_PLAN_CONF | sed 's|.mp3||' | tr '\n' ' '`
	
	
	run_all
	#run_features_manually_added
	#run_testset_manually_added_KV   # also edit AUDIO_FILES_PLAN_CONF !

}


run_all() {
	print_debug_message "run_all() - start"

	# remove old run results
	rm -rf ${RUN_OUTPUT_DIR}*
	
	run_output="${RUN_OUTPUT_DIR}tmp.txt"

	# read feature plan and execute it
	cat $FEATURE_PLAN_CONF | grep -v "^#" | sed '/^$/d' | \
	while read feature_name selected_feature_dimension yaafe_call; do
	
		# start new run
		if [[ "$feature_name" == "[run"* ]]; then   
		
			run_output="${RUN_OUTPUT_DIR}`echo $feature_name | sed 's|\[||' | sed 's|\]||'`.txt"
		
			clean_up
			
			printf "\n\n\n" >> $run_output
			echo "***********************************************************************************************" >> $run_output
			echo "***********************************************************************************************" >> $run_output
			echo "$feature_name" >> $run_output
			
		# evaluate run
		elif [[ "$feature_name" == "[/run"* ]]; then
			echo >> $run_output
			generate_test_set #> /dev/null
			generate_train_set
			while read cls_name cls cls_options; do
				print_debug_message "classify_and_evaluate() - start"
				classify_and_evaluate "$cls_name" "$cls" "$cls_options" >> $run_output
				print_debug_message "classify_and_evaluate() - finished"
			done < $CLASSIFIER_PLAN_CONF
			echo "***********************************************************************************************" >> $run_output
			echo "***********************************************************************************************" >> $run_output
			
		# add features to run
		else
			echo "$feature_name: $yaafe_call" >> $run_output
			
			# check if the feature has been already computed and then copy it simply
			precomputed_feature_dir="${PRECOMPUTED_FEATURE_FILES_DIR}${feature_name}"
			if [[ -d "$precomputed_feature_dir" ]]; then
				
				# make check that the precomputed feature contains all dimensions (i.e. has not been sliced yet)
				# this is done by checking if the first line is our header line starting with the TIME_LABEL_INDICATOR
				first_file="${precomputed_feature_dir}/`ls ${precomputed_feature_dir} | head -n 1`"
				first_line=$(head -n 1 $first_file)
				
				if  [[ $first_line == ${TIME_LABEL_INDICATOR}* ]] ; then
					echo "WARNING: The precomputed feature $precomputed_feature_dir does not contain all dimensions! The feature will be freshly calculated."
					generate_features "$feature_name" "$yaafe_call"
				else
					cp -R $precomputed_feature_dir $FEATURE_FILES_DIR
				fi				
			else
				generate_features "$feature_name" "$yaafe_call"
			fi

			# remove all non defined feature dimensions					
			yaafe_output_dir="${FEATURE_FILES_DIR}${feature_name}/"
					
			for hour in $HOURS; do
				feature_file_of_hour="${yaafe_output_dir}${hour}.${feature_name}"
				
				# if everything from the feature dimension should be taken then expand the "ALL" value to "1,2,..."
				if [[ "$selected_feature_dimension" == "ALL" ]]; then
					
					dimension=`head -1 $feature_file_of_hour | sed 's/[^,]//g' | wc -c`
					
					# the first column contains the time therefore count only to dimension-1 - this will be added later
					selected_feature_dimension="1"
					if [ "$dimension" -gt 2 ]; then
						for i in `seq 2 $((dimension-1))`; do
            				selected_feature_dimension="${selected_feature_dimension},${i}"
        				done
					fi
				fi
				
				# add label of first column which contains the time
				# marking the first colum with the indicator should help preventing accidently copying
				# pre-computed features that are already cut
				selected_feature_dimension_with_indicator="${TIME_LABEL_INDICATOR},${selected_feature_dimension}"
				
				# add to each index +1 because the cut command starts with 1 and not 0
				# make newlines the only separator
				old_IFS=$IFS
				IFS=$','
				selected_feature_dimension_for_cut="1"
				for j in $selected_feature_dimension; do
   					selected_feature_dimension_for_cut="${selected_feature_dimension_for_cut},$((j+1))"
				done
				IFS=$old_IFS
				
				# add header and slice the dimensions
				echo $selected_feature_dimension_with_indicator > "${feature_file_of_hour}.tmp"
				cat $feature_file_of_hour | cut -f$selected_feature_dimension_for_cut -d',' >> "${feature_file_of_hour}.tmp"
				mv "${feature_file_of_hour}.tmp" "${feature_file_of_hour}"
			done
		fi
	done
	
	print_debug_message "run_all() - finished"
}


run_features_manually_added() {
	# for manually adding feature files
	# do not forget to add the header on each feature file indicating the dimension/columns in there ("TIME_LABEL_INDICATOR,1,2,3,...")
	rm -rf ${TESTSET_DIR}*
	rm -rf ${TRAINSET_DIR}*
	rm -rf ${TMP_DIR}*
	rm -rf ${WEKA_MODEL_DIR}*
	rm -rf ${WEKA_TRAIN_OUTPUT_DIR}*
	rm -rf ${WEKA_PRED_DIR}*
	rm -rf ${LABELS_PRED_DIR}*
	generate_test_set #> /dev/null
	generate_train_set
	while read cls_name cls cls_options; do
		classify_and_evaluate "$cls_name" "$cls" "$cls_options"
	done < $CLASSIFIER_PLAN_CONF
}


run_testset_manually_added_KV () {
	# for using files from the KV
	# copy manuelly arff files from KV to testset-dir and choose other HOURS
	# do not forget to add the header on each feature file indicating the dimension/columns in there ("TIME_LABEL_INDICATOR,1,2,3,...")
	rm -rf ${TRAINSET_DIR}*
	rm -rf ${TMP_DIR}*
	rm -rf ${WEKA_MODEL_DIR}*
	rm -rf ${WEKA_TRAIN_OUTPUT_DIR}*
	rm -rf ${WEKA_PRED_DIR}*
	rm -rf ${LABELS_PRED_DIR}*
	generate_train_set
	while read cls_name cls cls_options; do
		classify_and_evaluate "$cls_name" "$cls" "$cls_options"
	done < $CLASSIFIER_PLAN_CONF
}


clean_up() {
	rm -rf ${FEATURE_FILES_DIR}*
	rm -rf ${TESTSET_DIR}*
	rm -rf ${TRAINSET_DIR}*
	rm -rf ${TMP_DIR}*
	rm -rf ${WEKA_MODEL_DIR}*
	rm -rf ${WEKA_TRAIN_OUTPUT_DIR}*
	rm -rf ${WEKA_PRED_DIR}*
	rm -rf ${LABELS_PRED_DIR}*
}


generate_features() {
	print_debug_message "generate_features() - start"

	yaafe_feature_name="${1}"
	yaafe_feature="${2}"
	yaafe_tmp_output_dir="${TMP_DIR}yaafe/"
	yaafe_output_dir="${FEATURE_FILES_DIR}${yaafe_feature_name}/"

	# move to audio directory, otherwise yaafe creates directory hierachy in output directory
	old_wd=`$PWD_PROG`
	cd $AUDIO_DIR
	
	# remove old outputs
	rm -rf $yaafe_tmp_output_dir

	# create directories for the new output
	mkdir $yaafe_tmp_output_dir
	mkdir $yaafe_output_dir
	
	# calculate features
	$YAAFE_PROG \
		-r 22050 \
		-f "${yaafe_feature_name}: ${yaafe_feature}" \
		-i $AUDIO_FILES_PLAN_CONF \
		-o csv -p Precision=8 -p Metadata=False \
		-b $yaafe_tmp_output_dir \
		&> "${TMP_DIR}yaafe_${yaafe_feature_name}.log"

	# convert the output of yaafe to the format that is used here
	# FROM: <value1>,...,<valuex>
	# TO:   <time>,<value1>,...,<valuex>
	for f in ${yaafe_tmp_output_dir}*; do

		f_conv=`basename "${f}" | sed "s|.mp3||" | sed "s|.csv||"`
		FEATURE_FILE_LEN=`cat $f | wc -l | bc`

		$PYTHON_PROG $PRINT_EVENLY_SPACED_NUMBERS_PROG \
			$AUDIO_LEN_SEC $FEATURE_FILE_LEN \
			| paste -d"," - $f \
			> "${yaafe_output_dir}${f_conv}"
	done

	# remove temp outputs
	rm -rf $yaafe_tmp_output_dir
	
	cd $old_wd
	
	print_debug_message "generate_features() - finished"
}


generate_test_set() {
	print_debug_message "generate_test_set() - start"
	
	first_hour=`echo $HOURS | cut -f 1 -d " "`
	num_of_features=`ls ${FEATURE_FILES_DIR}*/${first_hour}.* | wc -l | bc`
	for hour in $HOURS; do
		
		# check if the number of feature file is the same for each hour
		num=`ls ${FEATURE_FILES_DIR}*/${hour}.* | wc -l | bc`
		if [ $num_of_features != $num ]; then
			echo "[${0}] Number of feature files for hour: '${hour}' are different from the other hours. Add missing feature files."
			exit 1
		fi

		# list of all the feature files of the hour in the subdirectories
		feature_files=`ls ${FEATURE_FILES_DIR}*/${hour}.*`
	
		$PYTHON_PROG $LABEL_TO_ARFF_PROG \
			$STEP_SIZE \
			"${LABELS_DIR}${hour}.label" \
			"${TESTSET_DIR}${hour}.test.arff" \
			$feature_files #\
#			> /dev/null
	
	done
	
	print_debug_message "generate_test_set() - finished"
}


select_feature_dimension() {
	print_debug_message "select_feature_dimension() - start"
	
	print "TODO: implement"
	
	
	print_debug_message "select_feature_dimension() - finished"
}


generate_train_set() {
	print_debug_message "generate_train_set() - start"

	# for each hour (contains entries from the other hours except itself and its broadcasting station)

	for i in $HOURS; do
	
		train_file="${TRAINSET_DIR}${i}.train.arff"
		i_station=`echo $i | sed 's|-.*$||g'` # extract name of station
	
		# copy header from its own test set
		grep '^@' "${TESTSET_DIR}${i}.test.arff" > $train_file
		
		# copy instances from all other test sets except itself and its broadcasting station
		test_files=""
		for j in $HOURS; do
			j_station=`echo $j | sed 's|-.*$||g'`
			if [[ "$i" != "$j" ]]; then
			    if [[ "$USE_OWN_STATION_TRAINSET" == 1 ]] || [ "$i_station" != "$j_station" ]; then
				    test_files="$test_files ${TESTSET_DIR}${j}.test.arff"
				fi
			fi
		done		
		
		$PYTHON_PROG $GENERATE_TRAINSET_PROG \
			$NUM_OF_INSTANCES_PER_FILE \
			$TRAINSET_PERCENT_SPLIT \
			$train_file \
			$test_files

	done
	
	print_debug_message "generate_train_set() - finished"
}


classify_and_evaluate() {

	for hour in $HOURS; do
		
		classifier_name="${1}" 		#"RandomForest" #for filename
		classifier="${2}" 			#"weka.classifiers.trees.RandomForest"		
		classifier_params="${3}" 	#"-I 10 -K 4"
    
		trainset="${TRAINSET_DIR}${hour}.train.arff"
		testset="${TESTSET_DIR}${hour}.test.arff"
		model="${WEKA_MODEL_DIR}${hour}.${classifier_name}-model"
		train_output="${WEKA_TRAIN_OUTPUT_DIR}${hour}.${classifier_name}-out"
		prediction="${WEKA_PRED_DIR}${hour}.${classifier_name}-pred"
		
		# train model and evaluate on testset (also print the trees)
		"$JAVA_PROG" -mx4096m \
			$classifier $classifier_params \
			-t $trainset \
			-T $testset \
			-d $model \
			-print \
			> $train_output

		# write predictions to file
		"$JAVA_PROG" -mx4096m \
			$classifier \
			-l $model \
			-T $testset \
			-p 0 \
			> $prediction
		
		# remove all unnecassery stuff for statistic computation
		cat $prediction \
			| sed -e 's|^[ \t]*||' | grep ":" | cut -f2- -d' ' \
			| sed -e 's|+||g' | tr -s ' ' ' ' \
			| sed -e 's|:non_commercial||g' | sed -e 's|:commercial||g' \
			> "${prediction}-short"
			
	done
	
	# print top N features of the RandomForest
    output_files=`ls ${WEKA_TRAIN_OUTPUT_DIR}*.${classifier_name}-out`   
    $PYTHON_PROG $RF_TOP_N_FEATURES_PROG $output_files
	
    # print statistics
    prediction_files=`ls ${WEKA_PRED_DIR}*.${classifier_name}-pred-short`  
    for l in "${SMOOTHING_WINDOW_LENGTHS[@]}"; do
    	for i in "${SMOOTHING_ITERATIONS[@]}"; do
			labels_pred_subdir="${LABELS_PRED_DIR}/${classifier_name}_${l}_${i}/"
				mkdir -p $labels_pred_subdir
   				$PYTHON_PROG $CALC_STATISTICS_PROG "$classifier_name $classifier_params" \
   					"$l" "$i" "$SMOOTHING_VARIABLE_WINDOW" \
   					"$AUDIO_LEN_SEC" "$labels_pred_subdir" $prediction_files
		done
	done
}

print_debug_message() {
	if $PRINT_DEBUG ; then
		echo ${1}
	fi
}


time main "$@"
