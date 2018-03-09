
# coding: utf-8

# In[1]:

# Common Imports

#Pandas for creating dataframes
import pandas as pd

#Pyshark to capture packets
import pyshark

# Ploting
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

def remove_dict_key(layer_keys, layer_dict):
    for key in layer_keys:
        if key in layer_dict:
            del layer_dict[key]


# In[3]:

def get_dict_for_keys(layer_keys, layer_dict):
    new_dict = dict()
    for key in layer_keys:
        if key in layer_dict:
            new_dict[key] = layer_dict[key]
    return new_dict


# In[4]:

d = {x:x**2 for x in range(10)}
ks = [1,3,5]
{key:value for key, value in d.items() if key in ks}


# In[5]:

required_keys = ['ip.dst', 'ip.proto', 'tcp.flags.syn', 'tcp.flags.ack']


# In[6]:

# Reading packets from pre-captured file
file_cap = pyshark.FileCapture('captures/nov30_normal.pcapng')


# In[7]:

#!pip list


# In[8]:

print(file_cap[0].sniff_timestamp)
print(type(file_cap[0].sniff_timestamp))
print(float(file_cap[0].sniff_timestamp))
#print(file_cap[0].eth)

file_cap[0].sniff_time, 
file_cap[0].frame_info
# Hight level attributes
# dns_1.captured_length     dns_1.highest1_layer       dns_1.length              dns_1.transport_layer
# dns_1.dns                 dns_1.interface_captured  dns_1.pretty_print        dns_1.udp
# dns_1.eth                 dns_1.ip                  dns_1.sniff_time
# dns_1.frame_info          dns_1.layers              dns_1.sniff_timestamp


# In[19]:

get_ipython().magic('time')
# Converting capture file into dataframe for operations
#List of dataframes to hold
dfList = []
#file_cap.set_debug()
startTime = 0.0
endTime = 0.0
i = -1
first = True
while(True):
    i += 1
#     if i%100 == 0:
#         print(i)
    #Each packet can have multiple layers. Put them in list
    layerList = []
    #pyshark not able to handle AttributeError, AssertionError and KeyError(index values)
    try:
        #Iterate through layer of packet
        for layer in file_cap[i]:
            # Slice Data according to time
            t = float(file_cap[i].sniff_timestamp)
            #print(t)
            if startTime == 0.0:
                startTime = t
                endTime = startTime + 300
                testSetNo = 1
                print(testSetNo, startTime, endTime)
            elif t > endTime:
                #print(dfList)
                dfList.to_csv('t_set'+str(testSetNo))
                dfList = []
                startTime = endTime
                endTime = startTime + 300
                testSetNo += 1
                first = True
                print(testSetNo, startTime, endTime)
            #print(layer._layer_name)
            #We need only ip layer
            if layer._layer_name == 'ip':
                #Layer values are in the form of dictionary. Remove empty keys
                layer_dict = {key:value for key, value in layer._all_fields.items() if key in required_keys}
                #Filter the attributes.
                #layer_dict = {key:value for key, value in layer._all_fields.items() if key in required_keys}
                #Create dataframe from dictionary
                layerList.append(pd.DataFrame(layer_dict, index=[0]))
                #Add timestamp
                layerList.append(pd.DataFrame({'sniff_timestamp':file_cap[i].sniff_timestamp}, index=[0]))
                layerList.append(pd.DataFrame({'test_set':testSetNo}, index=[0]))
                #Build packet dataframe from layer frames. Its single row dataframe
                cDf = pd.concat(layerList, axis=1);
                #print(cDf)
                #print(cDf.columns)
                if first:
                    dfList = cDf
                    first = False
                #dfList = dfList.
                #dfList.append(pd.concat(layerList, axis=1))
                else:
                    dfList = dfList.append(cDf, ignore_index=True) 
                #print(dfList)
    except (AttributeError, AssertionError) as e:
        #print("error", e)
        continue  #print('Ipv4 packet does not exist')
    except  KeyError:
        break;


# In[13]:

# Create big table from all packet dataframe which are per packet
first = True
test_set = 99
for d in dfList:
    if test_set != d['test_set'].values[0] and not first:
        tdf.to_csv('converted/test_set'+str(test_set))
        test_set = d['test_set'].values[0]
        first = True
    else:
        #Get first dataframe
        if first:
            tdf = d
            first = False
        else:
            #Append all packets
            tdf = tdf.append(d.loc[:,~d.columns.duplicated()], ignore_index=True)   


# In[14]:

tdf.head()


# In[ ]:

#tdf


# In[ ]:

(dfList[0]['test_set'].values[0] == 1)


# In[12]:

tdf.columns


# In[ ]:

#pd.to_datetime(tdf['sniff_timestamp'], unit='s').head()


# In[ ]:

# import datetime
# print(datetime.datetime.fromtimestamp(float('1312966900.47579')).strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:

#t = tdf[['ip.dst', 'ip.proto', 'tcp.flags.syn', 'tcp.flags.ack', 'sniff_timestamp', 'test_set']]


# In[ ]:

t = tdf[['ip.dst', 'ip.proto', 'sniff_timestamp', 'test_set']]


# In[ ]:

t = t[t['ip.dst'].notnull()]


# In[ ]:

#t.columns = ['ip', 'proto', 'syn', 'ack', 'time_stamp', 'test_set']


# In[ ]:

t.columns = ['ip', 'proto', 'time_stamp', 'test_set']


# In[ ]:

ip_mapping = dict()
i = 0
for ip in t['ip'].unique():
    ip_mapping[i] = ip
    i += 1


# In[ ]:

t['ip'].unique()


# In[ ]:

#ip_mapping


# In[ ]:

mapping_df = pd.DataFrame(list(ip_mapping.values()), index=ip_mapping.keys())


# In[ ]:

mapping_df.columns = ['ip']


# In[ ]:

mapping_df.index.name = 'mapped'


# In[ ]:

mapping_df = mapping_df.reset_index().set_index(['ip'])


# In[ ]:

mapping_df.loc['test'] = 11


# In[ ]:

mapping_df.head()


# In[ ]:

mapping_df[mapping_df['mapped'] == 2]


# In[ ]:

#mapping_df.loc['224.0.0.22'][0]


# In[ ]:

t['ipmap'] = t['ip'].apply(lambda x: mapping_df.loc[x][0])


# In[ ]:

t.head()


# In[ ]:

# SYN packets is:
#tcp.flags.syn==1 && tcp.flags.ack==0


# In[ ]:

df = t.groupby(['ipmap', 'proto']).size().unstack().fillna(0).astype(int)


# In[ ]:

df.head()


# In[ ]:

d = pd.read_csv('converted/test_set1', index_col=0)


# In[ ]:

d.head()


# In[ ]:

df.values.shape, df.index.shape


# In[ ]:

#df.values


# In[ ]:

mapping_created = False
df_list = []
for test_set in range(1,18,1):
    tdf = pd.read_csv('converted/test_set'+str(test_set), index_col=0)
    #Filter Columns
    t = tdf[['ip.dst', 'ip.proto', 'sniff_timestamp', 'test_set']]
    #Remove null destinations
    t = t[t['ip.dst'].notnull()]
    #Rename Columns
    t.columns = ['ip', 'proto', 'time_stamp', 'test_set']
    #Map ip address to integer number
    if mapping_created == False:
        ip_mapping = dict()
        i = 0
        for ip in t['ip'].unique():
            ip_mapping[i] = ip
            i += 1

        #Create Dataframe for mapping ip to int value
        mapping_df = pd.DataFrame(list(ip_mapping.values()), index=ip_mapping.keys())
        mapping_df.columns = ['ip']
        mapping_df.index.name = 'mapped'
        mapping_df = mapping_df.reset_index().set_index(['ip'])
        mapping_created = True
    else:
        for ip in t['ip'].unique():
            if ip not in mapping_df.index.values:
                mapping_df.loc[ip] = mapping_df['mapped'].max()+1
                
    #Convert ip address from original dataframe to mapping 
    t['ipmap'] = t['ip'].apply(lambda x: mapping_df.loc[x][0])
    #Get count for each ip
    df = t.groupby(['ipmap', 'proto']).size().unstack().fillna(0).astype(int)
    df_list.append(df)


# In[ ]:

df_con = pd.concat(df_list)
df_con.to_csv('converted/test_mat')
mapping_df.to_csv('converted/mapping')


# In[ ]:

m = pd.read_csv('converted/test_mat', index_col=0)


# In[ ]:

m.head()


# In[ ]:

m.plot(x='6.0', y='17.0',kind='hexbin')
#Plot shows that they are not related


# In[ ]:

plt.pcolor(m.astype(float).corr())


# In[ ]:

#m.values, m.index.values


# In[ ]:

X = m.values; y = m.index.values


# In[ ]:

X.shape, y.shape


# In[ ]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[ ]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


# In[ ]:

#Preprocessing the data
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
#Check the scaleing mean
print(scaler.mean_)
#Check the scale
print(scaler.scale_)
#Transform Traning data
X_trans = scaler.transform(X_train)


# In[ ]:

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X, y)


# In[ ]:

# from sklearn.naive_bayes import 
# clf = BernoulliNB()
# clf.fit(X_trans, y_train)


# In[ ]:

# from sklearn.svm import LinearSVC
# lsvc = LinearSVC()
# lsvc.fit(X_train, y_train) 


# In[ ]:

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_trans)


# In[ ]:

print(kmeans.labels_)
print(kmeans.cluster_centers_)


# In[ ]:

X_pred = kmeans.predict(X)


# In[ ]:

from sklearn.decomposition import PCA
pca = PCA(n_components=3).fit(X)
pca_d = pca.transform(X)
#pca_c = pca.transform(X)


# In[ ]:

# Set a 3 KMeans clustering
#kmeans = KMeans(n_clusters=3)
# Compute cluster centers and predict cluster indices
#X_clustered = kmeans.fit_predict(x_9d)

# Define our own color map
LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b'}
label_color = [LABEL_COLOR_MAP[l] for l in X_pred]

# Plot the scatter digram
plt.figure(figsize = (7,7))
plt.scatter(pca_d[:,0],pca_d[:,1], c= label_color, alpha=0.5) 
plt.show()


# In[ ]:

X_test_trans = scaler.transform(X_test)


# In[ ]:

clf.score(X_test_trans, y_test)


# In[ ]:

lsvc.score(X_test_trans, y_test)


# In[ ]:

clf.predict([[0,0,0,0]])


# In[ ]:

df.loc[23]


# In[ ]:

wireshark_attr_lst_recreated = np.genfromtxt('wireshark_attr_lst.csv', delimiter=',', dtype=str)


# In[ ]:

layr = dict()
l = wireshark_attr_lst_recreated.tolist()
for a in wireshark_attr_lst_recreated:
    key = a.split('.')[0]
    if key in layr.keys():
        layr[key] = np.append(layr[key], a)
    else:
        layr[key] = a


# In[ ]:

layr['http']


# In[ ]:

tdf[tdf['tcp.flags.syn'].notnull()][['tcp.flags.syn']]


# In[ ]:

t = tdf
#t['ip.dst'] = t['ip.dst'].astype("category")
dst = 'ip.dst'
infected  = '147.32.84.165'
t = t[(t[dst].notnull()) & (t[dst] == infected)]


# In[ ]:

#Get values of other attributes when destination has address
lst = ['dns.a',
'dns.cname',
'dns.count.add_rr',
'dns.count.answers',
'dns.count.auth_rr',
'dns.count.labels',
'dns.count.queries',
'dns.flags',
'dns.flags.authenticated',
'dns.flags.authoritative',
'dns.flags.checkdisable',
'dns.flags.opcode']

# for inspect in lst:
#     print(t[[dst, inspect]][t[inspect].notnull()])

# t[t[dst] == '147.32.84.165']#[dst].value_counts()
# #t[dst].value_counts()
# t.groupby([dst] + lst).size()
# for inspect in lst:
#     print(t[[dst, inspect]][t[inspect].notnull()])


# In[ ]:

#Read HTML tables
htmldata = pd.read_html('https://en.wikipedia.org/wiki/List_of_IP_protocol_numbers')
#Tables are loaded in list
htmldata = htmldata[0]
#First row has index values
htmldata.columns = htmldata.loc[0].values
#Drop unnecessary columns
htmldata.drop(['Hex','References/RFC'], axis=1, inplace=True)
#Now drop first row
htmldata = htmldata[1:]
#Set protocol number as index
htmldata.set_index('Protocol Number', inplace=True)
#Write table to CSV
htmldata.to_csv('List_of_IP_protocol_numbers.csv')

