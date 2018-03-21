
# coding: utf-8

# ### Library imports

# In[1]:

#Pandas for creating dataframes
import pandas as pd

#Pyshark to capture packets
import pyshark
#For file operarions
import os
import gc
import sys


# ### Define required attributed needed to extract from capture.

# In[2]:

required_keys = ['ip.dst', 'ip.proto', 'tcp.flags.syn', 'tcp.flags.ack']


# ### Reading packets from pre-captured file using Pyshark

# In[3]:

file_cap = pyshark.FileCapture('captures/botnet-capture-20110810-neris.pcap')


# ### Extraxt IP packets from the captured file for further processing

# In[ ]:

get_ipython().magic('time')
#Write extraction result to
base_directory = 'converted/test3/attack_samples/1/'
#Debugger for pyshark
#file_cap.set_debug()
#Attributes for sample size
startTime = 0.0
endTime = 0.0
i = -1
first = True
#Sample Number
init_sample = 1
dfList = []
while(True):
    i += 1
    #Each packet can have multiple layers (e.g eth, ip, tcp etc.). 
    #Combine them in list for a single packet.
    layerList = []
    #pyshark not able to handle AttributeError, AssertionError and KeyError(index values)
    try:
        #Iterate through layer of packet
        for layer in file_cap[i]:
            # Slice Data according to time
            t = float(file_cap[i].sniff_timestamp)
            #Initial setup
            if startTime == 0.0:
                startTime = t
                endTime = startTime + 300 # 300 sec(5 min) is the sample size
                sample = init_sample
                print(sample, startTime, endTime)
            # Write every sample to csv file
            elif t > endTime:
                if not os.path.exists(base_directory):
                    os.makedirs(base_directory)
                dfList.to_csv(base_directory+str(sample))
                #Clear list after writing this would save memory
                startTime = endTime
                endTime = startTime + 300
                sample += 1
                #Reset after writing to file
                first = True
                print(sample, startTime, endTime)
                gc.collect()
            #We need only ip layer only
            if layer._layer_name == 'ip':
                #Layer values are in the form of dictionary. Filter the attributes
                layer_dict = {key:value for key, value in layer._all_fields.items() 
                              if key in required_keys}
                #Create dataframe from dictionary
                layerList.append(pd.DataFrame(layer_dict, index=[0]))
                #Add timestamp and sample number
                layerList.append(pd.DataFrame({'sniff_timestamp':file_cap[i].sniff_timestamp}, 
                                              index=[0]))
                layerList.append(pd.DataFrame({'sample':sample}, index=[0]))
                #Build packet dataframe from layer frames. Its single row dataframe
                cDf = pd.concat(layerList, axis=1);
                if first:
                    dfList = cDf
                    first = False
                else:
                    dfList = dfList.append(cDf, ignore_index=True)
    except (AttributeError, AssertionError) as e:
        continue  #print('Ipv4 packet does not exist')
    except  KeyError:
        break;

#If sample data is not written to file because for frame size
if(sample == init_sample):
    dfList.to_csv(base_directory+str(init_sample))


# In[ ]:



