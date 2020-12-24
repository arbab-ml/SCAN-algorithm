
import librosa
import numpy as np
import matplotlib.pyplot as plt
path="/media/ausserver4/DATA/Code/experiments/SCAN ALGO/Thesis 24 dec/Thesis/unsampled Two Samples/BITHNAH_20201022_181038.wav"
signal, sr = librosa.load(path,sr=None)    #Explicitly Setting sr=None ensures original sampling preserved -- STOVF    
#Segmenting to a random value of 1 seconds (for initial experiemtation)
length= signal.shape[0]/sr 

window_size = int(sr/1) # Considering one second
window_jump=int(sr/1)

window_iterator = 0
moving_averages = np.empty((0))
equivalent_time=np.empty((0))
while window_iterator < (len(signal) - window_size + 1):
    this_window = np.abs(signal[window_iterator : window_iterator + window_size])
    window_average = np.sum(this_window) / window_size
    moving_averages=np.append(moving_averages,window_average)
    equivalent_time=np.append(equivalent_time, window_iterator/sr)
    window_iterator += window_jump
moving_averages_final=moving_averages[1:-1]
equivalent_time_final=equivalent_time[1:-1]
max_point_index=np.argmax(moving_averages_final)
max_point_in_time=equivalent_time_final[max_point_index]
print(max_point_in_time)
plt.plot(moving_averages)
plt.show()
#Considering the values which are half second before the point of max signals
start_value= max_point_in_time  
end_value=max_point_in_time+1
