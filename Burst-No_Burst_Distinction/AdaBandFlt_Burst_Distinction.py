import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def test_valid_window(window, test_level = 5):
    """
    window : the window in the signal that has to be tested
    
    This funtion test the window to insure that it doesn't contain the signal of interest (spike)
    """
    #non zero ?
    second = np.percentile(window, 2)
    thirtyth = np.percentile(window, 30)
    #print(str(second) + "\t" + str(thirtyth) + "\t" + str(second/thirtyth))
    if abs(second/thirtyth) < test_level : 
        return True
    else : 
        return False
    
def init_noise_levels(signal, fs, 
                      noise_window_size = 0.01,
                      required_valid_windows = 100,
                      old_noise_level_propagation = 0.8, 
                      test_level = 5,
                      estimator_type = "RMS",
                      percentile_value = 25):
    
    if estimator_type == "RMS":
        return init_noise_levels_RMS(signal, fs, 
                      noise_window_size = noise_window_size,
                      required_valid_windows = required_valid_windows,
                      old_noise_level_propagation = old_noise_level_propagation, 
                      test_level = test_level,
                      percentile_value = percentile_value)
        
    elif estimator_type == "MAD":
        return init_noise_levels_MAD(signal, fs, 
                      noise_window_size = noise_window_size,
                      required_valid_windows = required_valid_windows,
                      old_noise_level_propagation = old_noise_level_propagation, 
                      test_level = test_level,
                      percentile_value = percentile_value)
    
    else: return None
    
    
def init_noise_levels_RMS(signal, fs, 
                      noise_window_size = 0.01,
                      required_valid_windows = 100,
                      old_noise_level_propagation = 0.8, 
                      test_level = 5,
                      percentile_value = 25):
    
    nb_valid_windows = 0
    list_RMS = []
    noise_levels = []
    
    noise_level = -1
    
     
    #boucle en indice
#    for window_index in range(0,len(signal)-(len(signal)%int(fs*noise_window_size)),int(fs*noise_window_size)):
    for window_index in range(0,len(signal),int(fs*noise_window_size)):
        test = test_valid_window(signal.iloc[window_index: window_index + int(fs*noise_window_size)], test_level)
        if nb_valid_windows < required_valid_windows:
            if test == True :
                RMS = np.sqrt(np.mean(signal.iloc[window_index: window_index + int(fs*noise_window_size)]**2))
                list_RMS.append(RMS)
                nb_valid_windows += 1
            
            if nb_valid_windows == required_valid_windows:
                noise_level = np.percentile(list_RMS, percentile_value)
                for elm in range(0, window_index, int(fs*noise_window_size)):
                    noise_levels.append(noise_level)
                
        else :
            """if test == True:
                if (window + int(fs*noise_window_size)) > (len(signal)-1) :
                    N25 = np.percentile(abs(signal.iloc[window:]), 25)
                else :
                    N25 = np.percentile(abs(signal.iloc[window: window + int(fs*noise_window_size)]), 25)
                noise_level = old_noise_level_propagation*noise_level + (1-old_noise_level_propagation)*N25
            noise_levels.append(noise_level)"""
            if test == True:
                if (window_index + int(fs*noise_window_size)) > (len(signal)-1) :
                    RMS = np.sqrt(np.mean(signal.iloc[window_index:]**2))
                else :
                    RMS = np.sqrt(np.mean(signal.iloc[window_index: window_index + int(fs*noise_window_size)]**2))
                list_RMS.append(RMS)
                NX = np.percentile(list_RMS, percentile_value)
                new_noise_level = old_noise_level_propagation*noise_level + (1-old_noise_level_propagation)*NX
                noise_level = new_noise_level
            noise_levels.append(noise_level)
            
    #cas ou il n'y a pas eut 100 fenetres de bruit valides rencontrees
    if noise_level == -1:
        
        #cas ou aucune fenetre valide n'a ete rencontree
        if noise_levels == []:
            for elm in range(0, len(signal), int(fs*noise_window_size)):
                noise_levels.append(0)
            
        else:
            noise_level = np.percentile(list_RMS, percentile_value)
            for elm in range(0, len(signal), int(fs*noise_window_size)):
                noise_levels.append(noise_level)
        
    
    noise_levels.append(noise_level)        
    plt.figure()
    plt.plot(list_RMS)
    plt.xlabel('Time Windows')
    plt.title('RMS values')
    plt.grid(True)
                
    return noise_levels

def init_noise_levels_MAD(signal, fs, 
                      noise_window_size = 0.01,
                      required_valid_windows = 100,
                      old_noise_level_propagation = 0.8, 
                      test_level = 5,
                      percentile_value = 25):
    
    nb_valid_windows = 0
    list_MAD = []
    noise_levels = []
     
    #boucle en indice
    for window_index in range(0,len(signal),int(fs*noise_window_size)):
        test = test_valid_window(signal.iloc[window_index: window_index + int(fs*noise_window_size)], test_level)
        if nb_valid_windows < required_valid_windows:
            if test == True :
                ###RMS = np.sqrt(np.mean(signal.iloc[window_index: window_index + int(fs*noise_window_size)]**2))
                MAD = scipy.stats.median_absolute_deviation(signal.iloc[window_index: window_index + int(fs*noise_window_size)])
                list_MAD.append(MAD)
                nb_valid_windows += 1
            
            if nb_valid_windows == required_valid_windows:
                noise_level = np.percentile(list_MAD, percentile_value)
                for elm in range(0, window_index, int(fs*noise_window_size)):
                    noise_levels.append(noise_level)
                
        else :
            """if test == True:
                if (window + int(fs*noise_window_size)) > (len(signal)-1) :
                    N25 = np.percentile(abs(signal.iloc[window:]), 25)
                else :
                    N25 = np.percentile(abs(signal.iloc[window: window + int(fs*noise_window_size)]), 25)
                noise_level = old_noise_level_propagation*noise_level + (1-old_noise_level_propagation)*N25
            noise_levels.append(noise_level)"""
            if test == True:
                if (window_index + int(fs*noise_window_size)) > (len(signal)-1) :
                    ###RMS = np.sqrt(np.mean(signal.iloc[window_index:]**2))
                    MAD = scipy.stats.median_absolute_deviation(signal.iloc[window_index:])
                else :
                    ###RMS = np.sqrt(np.mean(signal.iloc[window_index: window_index + int(fs*noise_window_size)]**2))
                    MAD = scipy.stats.median_absolute_deviation(signal.iloc[window_index: window_index + int(fs*noise_window_size)])
                list_MAD.append(MAD)
                NX = np.percentile(list_MAD, percentile_value)
                new_noise_level = old_noise_level_propagation*noise_level + (1-old_noise_level_propagation)*NX
                noise_level = new_noise_level
            noise_levels.append(noise_level)
    
    noise_levels.append(noise_levels)        
    plt.figure()
    plt.plot(list_MAD)
    plt.xlabel('Time Windows')
    plt.title('MAD values')
    plt.grid(True)
                
    return noise_levels



# find spikes

def find_spike(signal, initial_index, noise_levels, fs, spike_info,
               window_size = 0.002,
               noise_window_size = 0.01,
               threshold_factor = 3.5,
               maxseparation = 0.0008,
               time_checkmaxlocal = 0.0002,
               reduct_factor = 0.6,
               burst_threshold = 10): #facteur de réduction pour le threshold positif
    
    offset_index = int(np.round(signal.index[0]*fs/1000))
    
    if initial_index < len(signal) + offset_index:
        i = initial_index
        for value in signal.iloc[initial_index-offset_index:]:
            threshold = threshold_factor*noise_levels[int(np.round((i/fs)//noise_window_size))]
            if value < -threshold:

                #verifier qu'on est pas encore dans un effet du spike precedant et decaler la fenetre si besoin
                if signal.iloc[initial_index-offset_index-1] < -threshold:
                    for value_ in signal.iloc[initial_index-offset_index:]:
                        if value_ > -threshold:
                            return i
                        i += 1

                indice_1er_depass = i
                while(True):
                    if i < len(signal)+offset_index-1:
                        if signal.iloc[i-offset_index + 1]>signal.iloc[i-offset_index]:
                            break
                        else :
                            i+=1
                    else :
                        break
                             
                #partir à la recherche d'un max du spike a droite
                i_max_right = 'nan'  
                for k in range(int(np.round(maxseparation*fs))):
                    if (i-offset_index + k) < len(signal)-1:
                        if signal.iloc[i-offset_index+k] > reduct_factor*threshold and signal.iloc[i-offset_index+k]>signal.iloc[i-offset_index+k+1]:
                            if checkmaxlocal(signal, "right",i+k,offset_index,int(np.round(time_checkmaxlocal*fs))):
                                i_max_right = i+k
                                break
                #partir à la recherche d'un max du spike a gauche
                i_max_left = 'nan'  
                for k in range(int(np.round(maxseparation*fs))):
                    if (i-offset_index - k) > 0:
                        if signal.iloc[i-offset_index-k] > reduct_factor*threshold and signal.iloc[i-offset_index-k]>signal.iloc[i-offset_index-k-1]:
                            if checkmaxlocal(signal, "left",i-k,offset_index,int(np.round(time_checkmaxlocal*fs))):
                                i_max_left = i-k
                                break
                
                
                if i_max_left == 'nan' and i_max_right == 'nan':
                    #on a pas rencontré de spike
                    while signal.iloc[i-offset_index] < -threshold:
                        i += 1
                    return i
                
                else:
                    #récolte infos du spike
                    amplitude = 0
                    if i_max_left != 'nan' and i_max_right != 'nan':
                        if signal.iloc[i_max_left-offset_index] < signal.iloc[i_max_right-offset_index]:
                            amplitude = signal.iloc[i_max_right-offset_index] - signal.iloc[i-offset_index]
                        else :
                            amplitude = signal.iloc[i_max_left-offset_index] - signal.iloc[i-offset_index]
                    elif i_max_left != 'nan':
                            amplitude = signal.iloc[i_max_left-offset_index] - signal.iloc[i-offset_index]
                    elif i_max_right != 'nan':
                            amplitude = signal.iloc[i_max_right-offset_index] - signal.iloc[i-offset_index]
                    
                    if amplitude >= 2*burst_threshold:
                        is_it_burst = True
                    elif amplitude < 2*burst_threshold:
                        is_it_burst = False
  
                        # indice min, indice 1er depasssement
                        # max gauche, max droite
                        # variation d'amplitude entre min et max
                    
                    spike_info.append([i, indice_1er_depass,
                                        i_max_left, i_max_right,
                                        amplitude,is_it_burst])
                    return indice_1er_depass+int(np.round(window_size*fs))
                
                break  
            i += 1

    return -44

def checkmaxlocal(local_signal, sens, supposed_i_min,offset_index, nb_index_research=3):
    if(sens == "right"):
        k = 0
        while k <= nb_index_research:
        #for k in range(nb_index_research):
            if((local_signal.iloc[supposed_i_min-offset_index + k]) < (local_signal.iloc[supposed_i_min-offset_index + k + 1])):
                return False
            k += 1
        return True
    elif(sens == "left"):
        k = 0
        while k <= nb_index_research:
        #for k in range(nb_index_research):
            if((local_signal.iloc[supposed_i_min-offset_index - k]) < (local_signal.iloc[supposed_i_min-offset_index - k - 1])):
                return False
            k+=1
        return True
    else:
        return False

def find_spikes(signal, noise_levels, fs,
               window_size = 0.002,
               noise_window_size = 0.01,
               threshold_factor = 3.5,
               maxseparation = 0.0008,
               time_checkmaxlocal = 0.0002,
               burst_threshold = 10):
    
    initial = int(np.round(signal.index[0]*fs/1000))
    spike_info = []
    
    while initial != -44:
        initial = find_spike(signal, initial, noise_levels, fs, spike_info,
                             window_size = window_size,
                             noise_window_size = noise_window_size,
                             threshold_factor = threshold_factor,
                             maxseparation = maxseparation,
                             time_checkmaxlocal = time_checkmaxlocal,
                             burst_threshold = burst_threshold)

    df_spike_info = pd.DataFrame(spike_info)
    df_spike_info.columns = ['indice_min', 'indice_1er_depass','indice_max_gauche','indice_max_droite','Delta_amplitudes','burst?']

    return df_spike_info

#record spike

def record_spikes(signal, fs, spike_info,
                  align_method,
                  t_before = 0.001,
                  t_after = 0.002):
    
    if (align_method in spike_info.columns) == False:
        print("align_method is incorrect, please choose one of the following :" + str(spike_info.columns))
        return None
    
    else:
        spike_centers = spike_info[align_method].values
        
    t_b = int(np.round(fs*(t_before)))
    t_a = int(np.round(fs*(t_after)))
    
    data = np.array([[float(x) for x in range(t_b+t_a+1)]])
    
    initial_index = int(np.round(signal.index[0]*fs/1000))
    
    for center in spike_centers:
        if center < t_b + initial_index:
            spike = [0 for i in range(0, t_b-(center-initial_index))]
            spike = np.concatenate((spike, signal.values[:center + t_a - initial_index]))
            data = np.insert(data, len(data), spike, axis=0)
            
        elif center > len(signal)-t_a + initial_index:
            spike = signal.values[center - t_b - initial_index:]
            spike = np.concatenate((spike,[0 for i in range(0, t_a - (len(signal) + initial_index-center))]))
            data = np.insert(data, len(data), spike, axis=0)
            
        else :
            spike = signal.values[center - t_b - initial_index: center + t_a + 1 - initial_index]
            data = np.insert(data, len(data), spike, axis=0)

    print(np.shape(data))
    data = data.transpose()
    spike_data = pd.DataFrame(data)
    
    return spike_data

def record_spikes_oneline(signal, fs, spike_info,
                  align_method,
                  t_before = 0.001,
                  t_after = 0.002):

    if (align_method in spike_info.columns) == False:
        print("align_method is incorrect, please choose one of the following :" + str(spike_info.columns))
        return None
    
    else:
        spike_centers = spike_info[align_method].values
        
    offset_index = int(np.round(signal.index[0]*fs/1000))
    
    t_b = int(np.round(fs*(t_before)))
    t_a = int(np.round(fs*(t_after)))
    
    data = np.array(['NaN' for x in range(len(signal))])
    data = data.astype(float)
    times = np.array(['NaN' for x in range(len(signal))])
    times = times.astype(pd.Timestamp)
    
    for center in spike_centers:
        if center < t_b + offset_index:
            data[:center + t_a - offset_index] = signal.values[:center + t_a - offset_index]
            times[:center + t_a - offset_index] = signal.index[:center + t_a - offset_index]
            
        elif center > len(signal) - t_a + offset_index:
            data[center - t_b - offset_index:] = signal.values[center - t_b - offset_index:]
            times[center - t_b - offset_index:] = signal.index[center - t_b - offset_index:]
            
        else :
            data[center - t_b - offset_index: center + t_a + 1 - offset_index] = signal.values[center - t_b - offset_index: center + t_a + 1 - offset_index]
            times[center - t_b - offset_index: center + t_a + 1 - offset_index] = signal.index[center - t_b - offset_index: center + t_a + 1 - offset_index]

    spike_data_oneline = pd.DataFrame(data, index = times.astype(float))
    
    return spike_data_oneline

from random import randint

def print_spikes(spike_data,
                 t_before_alignement = 0,
                 first_spike = 1,
                 last_spike = -1,
                 fs = 25000,
                 randomize = False,
                 nb_spike = 20,
                 y_lim_min = -50,
                 y_lim_max = 60):
    
    if randomize == True:        
        kept = []
        m = len(spike_data.values[0])
        if m <= nb_spike:
            kept = [i for i in range(m)]
        else:      
            i = 0  
            while i < nb_spike:
                r = randint(0,m-1)
                if (r in kept) == False:
                    kept.append(r)
                    i += 1
        
        x = spike_data.iloc[:,kept].values
        
    else:
        x = spike_data.iloc[:,first_spike:last_spike]
        
    figure = plt.figure()
    t_b = int(np.round(fs*(t_before_alignement)))
    axes = figure.add_subplot(1, 1, 1)
    axes.plot((spike_data.iloc[:,0]-t_b)*1000/fs, x)
    axes.set_xlabel('Time in ms')
    axes.set_ylim(y_lim_min , y_lim_max)
    axes.grid()
    
"""
print_spikes(spike_data,
             t_before_alignement = 0.0015,
             first_spike = 1,
             last_spike = 20,
             fs = 25000,
             y_lim_min = -50,
             y_lim_max = 60)
             
print_spikes(spike_data,
             t_before_alignement = 0.0015,
             fs = 25000,
             randomize = True,
             nb_spike = 20,
             y_lim_min = -50,
             y_lim_max = 60)
"""


def plot_spikes_on_full_signal(signal):
    plt.plot(df.index, signal, color = 'blue')
    plt.plot(spike_data_oneline.index, spike_data_oneline, color = 'red')
    plt.title('Filtered Signal with Detected Spikes with RMS')
    plt.xlabel('Time Windows')
    plt.ylabel('Amplitude [µV]')
    plt.legend()
    plt.grid(True)