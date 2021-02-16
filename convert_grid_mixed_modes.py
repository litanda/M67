#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import os
import math
import matplotlib.pyplot as plt
import glob
import csv
from pathlib import Path
import pandas as pd
import re


# In[19]:


def read_history_file(histroyfile):
        with open(histroyfile) as f:
            content = [line.split() for line in f]

        # header of history file
        header = {content[1][i]:float(content[2][i]) for i in range(1,5)}

        # track of history file
        numberOfRows = len(content)-6
        numberOfCols = len(content[5])
        trackBlock = content[6:]
        colNames = content[5]
        formats = tuple([np.int32]+[np.float64 for i in range(numberOfCols)])
        track = np.zeros((numberOfRows),{'names':tuple(colNames),'formats':tuple(formats)})
        for irow in range(numberOfRows):
            track[irow]=tuple(trackBlock[irow])

        header = header
        track = track

        return header, track


# In[20]:


def extract_l_n(mode):
#extract l&n
    find = re.compile('-?\d+')
    xxx = find.findall(mode)
    output_l = float(xxx[0])
    output_n = float(xxx[1])
    return output_l, output_n


# In[21]:


###set up control parameter
#
###set up constant
###set up dr
datadr =  '/media/tanda/expansion/BNU-SYD-Model-Database/m67/'
outputdr = '/media/tanda/expansion//simple_grid_mixed_modes'


# In[22]:


###set up outputs
###Frequencies you want in the output list, shown as 'nu_l_n'
#Interpolation for a mode has to be same n. For mixed modes n = n_p - n_g
nu_l = [0,1,2]
nu_0_n_number = [i for i in range(1,41)]
nu_0_n_text = ['nu_0_%s' % s for s in nu_0_n_number]
E_0_n_text = ['E_0_%s' % s for s in nu_0_n_number]
nu_1_n_number = [i for i in range(-1000,41)]
nu_1_n_text = ['nu_1_%s' % s for s in nu_1_n_number]
E_1_n_text = ['E_1_%s' % s for s in nu_1_n_number]
nu_2_n_number = [i for i in range(-1000,41)]
nu_2_n_text = ['nu_2_%s' % s for s in nu_2_n_number]
E_2_n_text = ['E_2_%s' % s for s in nu_2_n_number]
output_freq_header = nu_0_n_text + nu_1_n_text + nu_2_n_text
output_E_hearder = E_0_n_text + E_1_n_text + E_2_n_text
####set up outputs file###
output_initial_header = ['initial_mass','initial_Z']
output_hist_header = ['model_number',
                      'star_age',
                      'star_mass',
                      'log_dt',
                      'cz_bot_radius',
                      'cz_csound',
                      'cz_scale_height',
                      'he_core_mass',
                      'effective_T',
                      'luminosity',
                      'radius',
                      'log_g',
                      'log_center_T',
                      'log_center_Rho',
                      'center_h1',
                      'center_he4',
                      'surface_h1',
                      'surface_he4',
                      'delta_nu',
                      'delta_Pg',
                      'nu_max',
                      'acoustic_cutoff']
output_all_header = output_initial_header + output_hist_header + output_freq_header
#output_header


# In[27]:


############################################################################################
###read and writ##########
os.chdir(datadr)
for folder in glob.glob("*/m*z*"):
    print(folder)
    #extract feh
    temp = re.search('/m(.+?)z', folder)
    if temp:
        inputmass = (temp.group(1))
    temp = re.search('y(.+?)a', folder)
    if temp:
        inputyinit = (temp.group(1))
    temp = re.search('a(.+?)fe', folder)
    if temp:
        inputMLT = (temp.group(1))

    inputfeh = '0'
    
    print(inputmass, inputyinit, inputfeh, inputMLT)
    outputname = outputdr + '/' + 'M'+ inputmass +'Y'+ inputyinit+'FeH'+ inputfeh +'a'+ inputMLT + '.csv'
    
    print(outputname)
        
    if (os.path.isfile(outputname)):
        continue
    filename = datadr + '/' + folder + '/history.data'
    if os.path.isfile(filename):
        header, track = read_history_file(filename)
        df = pd.DataFrame(track[output_hist_header])
        df.insert(0, 'initial_mass', np.mean(header['initial_mass']))
        df.insert(1, 'initial_feh', float(inputfeh))
        df.insert(2, 'initial_Z', np.mean(header['initial_z']))
        df.insert(3, 'initial_MLT', float(inputMLT)/10.0)
        df.insert(4, 'initial_Yinit', float(inputyinit)/100.0)
        df.insert(5, 'initial_ov', 0.018)
        #extracting seismic_frequencies
        proindex = np.loadtxt(datadr + '/' + folder + '/profiles.index', skiprows=2, usecols = [0,1,2])
        proindex_model_numebr = proindex[:,0].astype(int)
        proindex_osc_number = proindex[:,2].astype(int)
        model_number = df['model_number']
        #set up np array for frequences and inertia
        n_freq = np.size(output_freq_header)
        n_models = np.size(model_number)
        print(n_freq, n_models)
        seismo_freqs = np.zeros((n_models,n_freq))
        seismo_inertia = np.zeros((n_models,n_freq))
        #
        for num in model_number:
            nindex_t = np.nonzero(model_number == num)
            nindex = nindex_t[0]
            if (len(nindex) > 1):
                nindex = [nindex[0]]
            osc_num = proindex_osc_number[np.where(proindex_model_numebr == num)]
            #print(num, nindex,len(osc_num))
            if (len(osc_num) != 1):
                continue
            oscfile = datadr + '/' + folder + '/profile' + str(osc_num[0])+ '.data.FGONG.osc'
            if (os.path.isfile(oscfile)):
                #print(oscfile)
                ###read oscfile
                data = np.loadtxt(oscfile,skiprows=6, usecols = [0,1,2,3,4,5])
                mode_l = data[:,0]
                mode_np = data[:,1]
                mode_ng = data[:,2]
                mode_n = mode_np - mode_ng
                mode_freq = data[:,3]
                mode_inertia = data[:,5]
                for modename in output_freq_header:
                    i_c = output_freq_header.index(modename)
                    output_l, output_n = extract_l_n(modename)
                    i_freq = np.where((mode_l == output_l) & (mode_n == output_n))
                    if (mode_freq[i_freq].size == 1):
                        seismo_freqs[nindex[0],i_c] = mode_freq[i_freq]
                        seismo_inertia[nindex[0],i_c] = mode_inertia[i_freq]
        df_freq = pd.DataFrame(seismo_freqs, columns = output_freq_header)
        df_inertia = pd.DataFrame(seismo_inertia, columns = output_E_hearder)
        df_new = pd.concat([df, df_freq, df_inertia], axis = 1)
        df_new.to_csv(outputname)
exit()

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
