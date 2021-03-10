# -*- coding: utf-8 -*-
"""
#Command for getting the same folders structure
#rsync -a -f"+ */" -f"- *" "/media/ausserver4/DATA/Bats audio records/" "/media/ausserver4/DATA/Code/experiments/audio data analysis/audio-clustering/all plots"
"""
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
#from python_speech_features import mfcc, logfbank
import librosa
# Basic Libraries
import pandas as pd
import numpy as np
import librosa.display
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats

import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from scipy.io.wavfile import read as read_wav
import array
import glob
from pathlib import Path
plt.ioff()#turning interactive plotting OFF

import cv2
import IPython.display as ipd

EPS = 1e-8
#Reference: https://www.kaggle.com/haqishen/augmentation-methods-for-audio
#! USE THIS AFTER APPLYING THE rsync COMMAND. 



def calc_fft(y, rate):
	n = len(y)
	freq = np.fft.rfftfreq(n, d=1/rate)
	Y = abs(np.fft.rfft(y)/n)
	return Y, freq


def plot_signal(signals, labels_flag=False, rate=44100):
	fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
								 sharey=True, figsize=(5,5))


	original_xticks=list(range(len(signals))) #original x ticks
	new_xticks = np.array([x / rate for x in original_xticks])
	axes.plot(  new_xticks, list(signals))
	if labels_flag==False:
		axes.get_xaxis().set_visible(False)
		axes.get_yaxis().set_visible(False)
	else:
		fig.suptitle('Time Series', size=16)
		axes.get_xaxis().set_visible(True)
		axes.get_yaxis().set_visible(True)
		axes.set_xlabel("Time in seconds")
		axes.set_ylabel("Amplitude")
	return fig
	
def plot_fft(fft, labels_flag=False):
	fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
								 sharey=True, figsize=(5,5))
	data = (fft)
	Y, freq = data[0], data[1]

	axes.plot(freq, Y)
	if labels_flag==False:
		axes.get_xaxis().set_visible(False)
		axes.get_yaxis().set_visible(False)
	else:
		fig.suptitle('FFT', size=16)
		axes.get_xaxis().set_visible(True)
		axes.get_yaxis().set_visible(True)
		axes.set_xlabel("Frequency")
		axes.set_ylabel("Amplitude")
	return fig

def plot_fbank(fbank, labels_flag=False):
	fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
								 sharey=True, figsize=(5,5))
	fig.suptitle('Filter bank coefficients', size=16)
	#axes.set_title(list(fbank.keys())[i])
	axes.imshow((fbank),cmap='hot', interpolation='nearest')
	if labels_flag==False:
		axes.get_xaxis().set_visible(False)
		axes.get_yaxis().set_visible(False)
	else:
		fig.suptitle('Filter bank coefficients', size=16)
		axes.get_xaxis().set_visible(True)
		axes.get_yaxis().set_visible(True)
		axes.set_xlabel("Frequency")
		axes.set_ylabel("Amplitude")
	return fig

def plot_mfcc(mfcc):
	fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
								 sharey=True, figsize=(5,5))
	fig.suptitle('Mel Frequency Cepstral Coefficients', size=16)
	#axes.set_title(list(mfccs.keys())[i])
	axes.imshow(mfcc,cmap='hot', interpolation='nearest')
	axes.get_xaxis().set_visible(False)
	axes.get_yaxis().set_visible(False)
	return fig

#Ref: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
def plot_spectrogram(signals, labels_flag=False, rate=44100,mode="simple", target_length=3):
	bottom_value=20000
	top_value=92000

	fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
								 sharey=True, figsize=(5,5))
	fig.suptitle('Spectrogram', size=16)
	#axes.set_title(list(fbank.keys())[i])
	hop_length_value=512
	if (mode=="simple"):
		spec = np.abs(librosa.stft(signals, hop_length=hop_length_value, n_fft=5120))
		spec = librosa.amplitude_to_db(spec, ref=np.max)
		axes=librosa.display.specshow(spec, sr=rate,hop_length=hop_length_value, x_axis='time', y_axis='linear') #when computing an STFT, you can pass that same parameter to specshow. 
																					#This ensures that axis scales (e.g. time or frequency) are computed correctly.
		ax2 = fig.gca() #getting this to set the range
		print("Before adjusting axes, the yaxis range was: ", ax2.get_ylim())
		ax2.set_ylim(bottom=bottom_value, top=top_value) # Setting frequency betweei 20KHz and 92KHz
		return fig

	elif (mode == "timeshift"):
		start_ = int(np.random.uniform(- (0.2*rate),(0.2*rate)))
		print('time shift: ',start_)
		if start_ >= 0:
			wav_time_shift = np.r_[signals[start_:], np.random.uniform(-0.001,0.001, start_)]
		else:
			wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), signals[:start_]]
		#^ processing done, now plotting
		spec = np.abs(librosa.stft(wav_time_shift, hop_length=hop_length_value, n_fft=5120))
		spec = librosa.amplitude_to_db(spec, ref=np.max)
		axes=librosa.display.specshow(spec, sr=rate, x_axis='time',hop_length=hop_length_value, y_axis='linear')
		ax2 = fig.gca() #getting this to set the range
		print("Before adjusting axes, the yaxis range was: ", ax2.get_ylim())
		ax2.set_ylim(bottom=bottom_value, top=top_value) # Setting frequency betweei 8KHz and 92KHz
		return fig

	elif (mode == "speedtune"):
		speed_rate = np.random.uniform(0.7,1.3)
		wav_speed_tune = cv2.resize(signals, (1, int(len(signals) * speed_rate))).squeeze()
		print('speed rate: %.3f' % speed_rate, '(lower is faster)')
		if len(wav_speed_tune) < (target_length*rate):
			pad_len = (target_length*rate) - len(wav_speed_tune)
			wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
								wav_speed_tune,
								np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
		else: 
			cut_len = len(wav_speed_tune) - (target_length*rate)
			wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+int(target_length*rate)]
		#^ processing done, now plotting
		spec = np.abs(librosa.stft(wav_speed_tune, hop_length=hop_length_value, n_fft=5120))
		spec = librosa.amplitude_to_db(spec, ref=np.max)
		axes=librosa.display.specshow(spec, sr=rate, x_axis='time',hop_length=hop_length_value, y_axis='linear')
		ax2 = fig.gca() #getting this to set the range
		print("Before adjusting axes, the yaxis range was: ", ax2.get_ylim())
		ax2.set_ylim(bottom=bottom_value, top=top_value) # Setting frequency betweei 8KHz and 92KHz
		return fig
	else:
		print("Wrong Mode Selected")

	


def plot_mel_spectrogram(signals, labels_flag=False, rate=44100):
	fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
								 sharey=True, figsize=(5,5))
	fig.suptitle('Mel Spectrogram', size=16)
	#axes.set_title(list(fbank.keys())[i])
	mel_spect = librosa.feature.melspectrogram(y=signals, sr=rate)
	mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
	axes=librosa.display.specshow(mel_spect,sr=rate, y_axis='mel', x_axis='time');

	return fig
#https://www.kaggle.com/haqishen/augmentation-methods-for-audio


mode='plotting'#"just-calculations"
sample_size=1000
#input_directory = r'/home/rabi/Documents/Thesis/unsampled Two Samples'
input_directory = r'/media/rabi/Data/ThesisData/Bats audio records'
#output_directory=r'/home/rabi/Documents/Thesis/audio data analysis/audio-clustering/all plots'
output_directory=r'/media/rabi/Data/ThesisData/audio data analysis/audio-clustering/plots_10march'

bat_calls_data_dir=r'/home/rabi/Documents/Thesis/batdetect/bat_eval/results/results.csv'

final_target_length=0.5 #1 second
iterator=0
random.seed(5) 
all_stats=pd.DataFrame(columns=["file_name", "length", "sample rate"])

bat_calls_data=pd.read_csv(bat_calls_data_dir)

bat_calls_data_processed=bat_calls_data.iloc[bat_calls_data.groupby('file_name')['detection_prob'].agg(pd.Series.idxmax)]
bat_calls_data_processed=bat_calls_data_processed.set_index("file_name")

for path in tqdm(Path(input_directory).rglob('*.wav')):
	try:
		save_to= str(path.relative_to(input_directory)).split('/')[-1]
		signal, sr = librosa.load(path,sr=None)    #Explicitly Setting sr=None ensures original sampling preserved -- STOVF    
		#Segmenting to a random value of 1 seconds (for initial experiemtation)
		length= signal.shape[0]/sr 
		temp_results={"file_name":save_to, "length": length, "sample rate": sr}    
		all_stats=all_stats.append(temp_results, ignore_index=True)
		
		iterator=iterator+1
		#print(iterator)
		if (iterator>=sample_size):
		    break
		
		if (mode=="just-calculations"):
			continue
		#choosing the random starting point
		if(length<final_target_length):
			print("lengh less than threshold")
			continue
			

		#Old, random segmentation
		#start_value=random.randrange(0, int(length)-final_target_length, 1)
		#end_value=start_value+final_target_length

		# Version 2.0 (Now old-dated)

				# #New, segmenting the values according to maximum moving average value
				# window_size = int(sr/1) # Considering one second
				# window_jump=int(sr/1)

				# window_iterator = 0
				# moving_averages = np.empty((0))
				# equivalent_time=np.empty((0))
				# while window_iterator < (len(signal) - window_size + 1):
				# 	this_window = np.abs(signal[window_iterator : window_iterator + window_size])
				# 	window_average = np.sum(this_window) / window_size
				# 	moving_averages=np.append(moving_averages,window_average)
				# 	equivalent_time=np.append(equivalent_time, window_iterator/sr)
				# 	window_iterator += window_jump
				# max_point_index=np.argmax(moving_averages)
				# max_point_in_time=equivalent_time[max_point_index]
				# print(max_point_in_time)
				
				# #Considering the values which are half second before the point of max signals
				# start_value= int(max_point_in_time  )
				# end_value=int(max_point_in_time+1)
		
		#Version 3.0 (using batdetect algorithm to select the segment from audio)

			#Now, spread the point of maximum confidence and spread start_value and end_value around it (total duration =0.5 seconds)
		
		local_file_path='/'+str(path.relative_to(input_directory))
		detected_time=bat_calls_data_processed.loc[local_file_path]['detection_time']
		start_value=(detected_time-(final_target_length/2))
		end_value=(detected_time+(final_target_length/2))
		
		signal=signal[int(start_value*sr):int(end_value*sr)]
		#Now saving the statistics (sr and length)
		#filename_short=save_to.split('/')[-1]
		#Now, storing the plots       
		# plot_signal(signal,labels_flag=True).savefig(output_directory+'/signals/'+save_to+'.png')
		# plot_fft(calc_fft(signal, sr),labels_flag=True).savefig(output_directory+'/ffts/'+save_to+'.png')
		# plot_spectrogram(signal).savefig(output_directory+'/spectrograms/'+save_to+'.png')
		#plot_mel_spectrogram(signal, rate=sr).savefig(output_directory+'/mel spectrograms/'+save_to+'.png')
		#Making the plots, 3 modes are available = simple, speedtune, timeshift
		new_length=len(signal)/sr
		plot_spectrogram(signal,rate=sr, mode="simple", target_length=new_length).savefig(output_directory+'/spectrograms/batsnet_train/1/'+save_to+'.png')
		#plot_spectrogram(signal,rate=sr, mode="simple", target_length=length).savefig(output_directory+'/spectrograms/original/'+save_to+'.png')
		plt.close()


		
		random_augmentation=random.randint(0, 1)
		if(random_augmentation==0):
			plot_spectrogram(signal,rate=sr, mode="timeshift", target_length=new_length).savefig(output_directory+'/spectrograms/augmented/'+save_to+'.png')
			plt.close()


		else:
			plot_spectrogram(signal,rate=sr, mode="speedtune",target_length=new_length).savefig(output_directory+'/spectrograms/augmented/'+save_to+'.png')
			plt.close()




		#break
	except:
		continue
	
##Removing extreme values in all_stats
exit(0)		

all_stats.to_csv("/home/rabi/Documents/Thesis/audio data analysis/all_stats.csv")



all_stats_df=all_stats.copy()[["length", "sample rate"]]
#Save this all_stats_df to a file. 
all_stats_final=all_stats_df[np.abs(all_stats_df.length-all_stats_df.length.mean()) <= (3*all_stats_df.length.std())]


##Plotting the Histograms
		
	
   #REference: https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0 
for i, binwidth in enumerate([1, 5, 10, 15]):
	
	# Set up the plot
	ax = plt.subplot(2, 2, i + 1)
	
	# Draw the plot
	ax.hist(all_stats_final['length'], bins = int(180/binwidth),
			 color = 'blue', edgecolor = 'black')
	
	# Title and labels
	ax.set_title('Histogram with Binwidth = %d' % binwidth)#, size = 30)
	ax.set_xlabel('Audio length (seconds)')#, size = 22)
	ax.set_ylabel('Count')#, size= 22)

plt.tight_layout()
plt.savefig("Histogram of Length.png")
plt.show()
	
	
	
	
	   #REference: https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0 
for i, binwidth in enumerate([1, 5, 10, 15]):
	
	# Set up the plot
	ax = plt.subplot(2, 2, i + 1)
	
	# Draw the plot
	ax.hist(all_stats_final['sample rate'], bins = int(180/binwidth),
			 color = 'blue', edgecolor = 'black')
	
	# Title and labels
	ax.set_title('Histogram with Binwidth = %d' % binwidth)#, size = 30)
	ax.set_xlabel('Sample Rate')#, size = 22)
	ax.set_ylabel('Count')#, size= 22)

plt.tight_layout()
plt.savefig("Histogram of Sampling Rate.png")
plt.show()
	
	
	
	

