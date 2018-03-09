
# coding: utf-8

# In[1]:

# Common Imports
#Pandas for creating dataframes
import pandas as pd
#Sklearn
from sklearn import preprocessing
#OS moduled for file oprations
import os
#CSV module
import csv
#SKlearn SVM
from sklearn import svm


# In[2]:

#Folder Base Path
base_path = 'converted/test2/'


# In[3]:

first = True

ip_dict = dict()
sample_path = base_path+'samples/'
#Cluster Path
#cluster_path = base_path+'attack_ip_cluster/2/'
sample_count = 1;
first = True;
dfList = []
for filename in os.listdir(sample_path):
    tdf = pd.read_csv(sample_path+filename, index_col=0)
    #Filter Columns
    tdf = tdf[['ip.dst', 'ip.proto', 'sniff_timestamp', 'sample']]
    #Remove null destinations
    tdf = tdf[tdf['ip.dst'].notnull()]
    #Rename Columns
    tdf.columns = ['ip', 'proto', 'time_stamp', 'sample']
    #Create data frame for each sample file contain IP address with packet counts
    df = tdf.groupby(['ip', 'proto']).size().unstack().fillna(0).astype(int)
    df['sample'] = filename
    #Create list of eac dataframe
    dfList.append(df)    


# In[5]:

#cobine all dataframes and sort by IP address
df1 = pd.concat(dfList).sort_index()
#drop old index and create new multi-index with IP address and sample so that all the sample data for a given IP is cobined.
df1 = df1.reset_index().set_index(['ip','sample'])


# In[7]:

df1.head()


# In[9]:

#Get list of all IP address
indexValues = df1.index.get_level_values(0)


# In[11]:

indexValues


# In[80]:

#Train SVM for each IP address data
svm_dict = dict()
for i in indexValues:
    X_train = df1.loc[i].values
    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    #Store trained SVM for each IP
    svm_dict[i] = clf


# In[90]:

#Predict for the give destination if it is normal or not
svm_dict['192.168.0.7'].predict([[0,0,1,1]])

