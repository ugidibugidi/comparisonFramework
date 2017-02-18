import numpy as np
import os.path
import sys


def printMessage(msg):
	print "[" + os.path.basename(__file__) + "] " + msg


def run():

	#statisticName = "RandomForest"
	#files = [ "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/ARD-h1.RandomForest-pred-short", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/ARD-h16.RandomForest-pred-short", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/RTL-h2.RandomForest-pred-short", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/RTL-h20.RandomForest-pred-short", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/Sat1-h1.RandomForest-pred-short", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/Sat1-h7.RandomForest-pred-short", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/ZDF-h2.RandomForest-pred-short", "/Users/balder/Documents/Uni/masterarbeit/data/weka/comparisonFramework/weka/predictions/ZDF-h17.RandomForest-pred-short" ]
	
	#mkdir _temp_pred_labels/
	#python calcStatistics_v2.py foo 32 2 true 3600.0 _temp_pred_labels/ `ls ../weka/predictions/*.RandomForest-pred-short`
	
	if len(sys.argv) < 8:
		print "Too few parameters."
		exit(1)
	
	statisticName = sys.argv[1]
	smoothingWindow = int(sys.argv[2]) # if < 0 then no smoothing
	numberOfIteration = int(sys.argv[3])
	variable_convolving = sys.argv[4] in ('1', 'true', 'True', 'TRUE') 
	lenInSec = float(sys.argv[5]) # total length of the audio file
	labelFolder = sys.argv[6]
	files = sys.argv[7:]
	
	
	printStatsTable(statisticName, files, labelFolder, lenInSec, useSmoothing=(smoothingWindow > 0), window_len=smoothingWindow, numberOfIteration=numberOfIteration, window='hanning', variable_convolving=variable_convolving)
	
	'''
	statisticName = "RandomForest -K 128 -I 512 -depth 0"
	i=1
	while i < 16385 :
		printStatsTable(statisticName, files, labelFolder, lenInSec, useSmoothing=True, window_len=i, window='hanning')
		i *= 2
	'''
	

def printStatsTable(statisticName, files, labelFolder, lenInSec, useSmoothing=True, window_len=1024, numberOfIteration=1, window='hanning',variable_convolving=False):
	print "========================================================================================="
	print "	" + statisticName
	if useSmoothing :
		print "	" + "Smoothing" + " (" + str(window) + "): window_length=" + str(window_len) + ", iterations=" + str(numberOfIteration) + ", variable_window=" + str(variable_convolving)
	else :
		print
	print "Filename	TP	FP	TN	FN	Acc	BaseAcc	Prec	Recall	F-Measure"
	print "-----------------------------------------------------------------------------------------"
	
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	
	for f in files :
		if ( (not os.path.isfile(f)) or os.stat(f).st_size == 0 ):
			printMessage("File does not exit or is empty: " + f)
		else :
			predictions = np.genfromtxt(f, delimiter=' ', usecols=(0,1), dtype=np.int)
			predictions_probability = np.genfromtxt(f, delimiter=' ', usecols=(2), dtype=np.float)
			
			# The prediction probability gives the probabilty of the predicted class which is always
			# larger 0.5 therefore subtract 1 for the non-commercials probability (and do not change it for 
			# commercials). Add 1 to both classes because 1 is counted as non-commercial and 2 is 
			# counted as commercial.
			normalized_predictions_probability = np.empty_like(predictions_probability)
			for i,e in enumerate(predictions_probability):
				# result: non-commercials are closer to 1
				if predictions[i][1] == 1:
					normalized_predictions_probability[i] = 1 + (1-e)
				# result: commercials are closer to 2
				else:
					normalized_predictions_probability[i] = 1 + e
			
			if useSmoothing == True:
				#p_smoothed = smooth(predictions[:,1],window_len,window)
				
				# Uncomment this line if the originial smoother should be used (that uses 1/2 as input instead
				# of the prediction probabilities (errors)
				#p_smoothed = smooth_multiple_times(predictions[:,1],window_len,numberOfIteration,window)
				p_smoothed = smooth_multiple_times(normalized_predictions_probability,window_len,numberOfIteration,window,variable_convolving)
				
				#for i,p in enumerate(predictions):
					#print str(i) + ": " + str(p) + " " + str((np.rint(p_smoothed[i])).astype(int))
				predictions[:,1] = (np.rint(p_smoothed)).astype(int)
			
			fTP = sum(1 for p in predictions if p[0] == 2 and p[1] == 2)
			fFP = sum(1 for p in predictions if p[0] == 1 and p[1] == 2)
			fTN = sum(1 for p in predictions if p[0] == 1 and p[1] == 1)
			fFN = sum(1 for p in predictions if p[0] == 2 and p[1] == 1)
		
			fname = (os.path.basename(f)).split(".")[0]
			printStats(fname, fTP, fFP, fTN, fFN)
			
			printLabels(predictions[:,1], lenInSec, labelFolder + fname + "_pred.label")
		
			TP += fTP
			FP += fFP
			TN += fTN
			FN += fFN
		
	print "-----------------------------------------------------------------------------------------"
	printStats("OVERALL", TP, FP, TN, FN)
	#printStats(str(window_len) + ": OVERALL", TP, FP, TN, FN)
	print "========================================================================================="


def printLabels(predictions, lenInSec, file) :
	nonCommLabel = "0;"
	commLabel = "2;"
	
	stepSize = lenInSec / len(predictions)
	
	with open(file, "w") as f:
		startPosition = 0.0
		duration = 0.0
		label = (nonCommLabel if predictions[0] == 1 else commLabel)
	
		for i in range(1,len(predictions)):
			if predictions[i] != predictions[i-1]:
				duration = (i * stepSize) - startPosition
				f.write(str(startPosition) + "," + str(duration) + "," + label + "\n")
				startPosition = i * stepSize
				label = (nonCommLabel if predictions[i] == 1 else commLabel)
	
		duration = lenInSec - startPosition
		label = (nonCommLabel if predictions[-1] == 1 else commLabel)
		f.write(str(startPosition) + "," + str(duration) +  "," + label + "\n")


def printStats(name, TP, FP, TN, FN):

	# accuracy
	ACC = 0.0
	if (TP + FP + TN + FN) > 0 :
		ACC = (TP + TN)*100.0 / (TP + FP + TN + FN)*1.0
		
	# calculate baseline accuracy
	positives = (TP + FN) * 1.0
	negatives = (TN + FP) * 1.0
	total = positives + negatives
	baselineACC = (positives if positives > negatives else negatives)*100.0 / total
	
	# f-measure
	PREC = TP*100.0 / (TP + FP)*1.0 if (TP + FP) > 0 else 0.0
	RECALL = TP*100.0 / (TP + FN)*1.0 if (TP + FN) > 0 else 0.0
	F_MEASURE = (2 * PREC * RECALL) / (PREC + RECALL) if (PREC + RECALL) > 0 else 0.0
		
	print name + "		" + \
		str(TP)  + "	" + \
		str(FP)  + "	" + \
		str(TN)  + "	" + \
		str(FN)  + "	" + \
		'{:>7}'.format('{0:.3f}'.format(ACC)) + "	" + \
		'{:>7}'.format('{0:.3f}'.format(baselineACC)) + "	" + \
		'{:>7}'.format('{0:.3f}'.format(PREC)) + "	" + \
		'{:>7}'.format('{0:.3f}'.format(RECALL)) + "	" + \
		'{:>7}'.format('{0:.3f}'.format(F_MEASURE))
		


# from: http://wiki.scipy.org/Cookbook/SignalSmooth
def smooth(x,window_len=1024,window='hanning',variable_convolving=False):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    if variable_convolving:
        y = variable_convolve(s,window_len,window)
    else:
        y = np.convolve(w/w.sum(),s,mode='valid')
    
	#make sure that it is the right length
    return (y[(window_len/2-1):-(window_len/2)])[:len(x)]


def smooth_multiple_times(x,window_len=1024,times=1,window='hanning',variable_convolving=False):
	"""smooth the data using a window with requested size and the number of 
	times the smoothing algorithm is performed on the signal.
	"""
	
	smoothed = x
	
	for i in xrange(0, times):
		smoothed = smooth(smoothed,window_len,window,variable_convolving)
    
	return smoothed


def variable_convolve(signal,max_window_len,window_function='hanning'):
	""" Convolve areas with many non-commercials (value is near 1) with shorter windows.
	Convolve areas with many commercials (value is near 2) with larger windows. This should help
	to keep short commercials because the window is small. Long commercials should be kept because 
	the window is larger.
	"""
	
	minimum_window_len = 3
	
	# Just for test output
	#temp_max = 2111

	if max_window_len < minimum_window_len:
		return signal
	
	convolved_signal = np.zeros_like(signal)
	
	for signal_i,s in enumerate(signal):
		
		lower_bound = max(0, (signal_i-(max_window_len/2)))
		upper_bound = min(signal.size, (signal_i+(max_window_len/2)))
		
		# calculate the average of the neighbors
		# -1 because it contains values between 1-2
		normalized_mean = np.mean(signal[lower_bound:upper_bound]) - 1
		
		actual_window_len = max(minimum_window_len, int(normalized_mean*max_window_len+1))
		
		if window_function == 'flat': #moving average
			window = np.ones(actual_window_len,'d')
		else:
			window = eval('np.'+window_function+'(actual_window_len)')
		
		normalized_window = window/window.sum()

		# convolve
		for window_i, w in enumerate(normalized_window):
			# do not care that the values do not sum up to 1 of the window at the beginning and end
			# because they will be removed anyway
			if (signal_i-window_i/2) >= 0 and (signal_i+window_i/2) < signal.size:
				convolved_signal[signal_i] = convolved_signal[signal_i] + signal[signal_i+(window_i-actual_window_len/2)]*w
	
		#if signal_i < temp_max:
		#	commercial_marker = "   "
		#	if signal[signal_i] >= 1.5:
		#		commercial_marker = " | "
		#	if convolved_signal[signal_i] >= 1.5:
		#		commercial_marker = commercial_marker + "|"	
		#	
		#	print '{0: 5d}'.format(int(signal_i)), " (", '{0: 4d}'.format(int(actual_window_len)), "): ", '{0:.3f}'.format(signal[signal_i]), " - ", '{0:.3f}'.format(convolved_signal[signal_i]), commercial_marker

	return convolved_signal





run()
