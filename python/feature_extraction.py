
# coding: utf-8

# In[2]:


# Common Imports

#Pandas for creating dataframes
import pandas as pd


# In[6]:


tdf = pd.read_csv('converted/s1', index_col=0)
#Filter Columns
t = tdf[['ip.dst', 'ip.proto', 'sniff_timestamp', 'sample']]
#Remove null destinations
t = t[t['ip.dst'].notnull()]
#Rename Columns
t.columns = ['ip', 'proto', 'time_stamp', 'sample']
#Get count for each ip
df = t.groupby(['ip', 'proto']).size().unstack().fillna(0).astype(int)


# In[87]:


#df


# In[12]:


X = df.values


# In[14]:


#Preprocessing the data (Feature Scaling)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
#Check the scaleing mean
print(scaler.mean_)
#Check the scale
print(scaler.scale_)
#Transform Traning data
X_trans = scaler.transform(X)


# In[21]:


X_trans.shape


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X_trans).score(X_trans) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In[56]:


#Define Number of Clusters
cluster_count = 6


# In[57]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=cluster_count)
kmeans.fit(X_trans)


# In[58]:


print(kmeans.labels_)
print(kmeans.cluster_centers_)


# In[81]:


ip_series = pd.Series(df.index, name='ip')


# In[82]:


label_series = pd.Series(kmeans.labels_, name='label')


# In[86]:


ip_label_df = pd.concat([ip_series, label_series], axis=1).set_index('ip')


# In[ ]:


ip_label_df.csv('converted/ip_label')


# In[70]:


X_pred = kmeans.predict([X[0]])


# In[71]:


X_pred


# In[60]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3).fit(X)
pca_d = pca.transform(X)
#pca_c = pca.transform(X)


# In[64]:


from matplotlib import colors
#Get colors
#color_list = list(colors.cnames.keys())
color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
# Define our own color map
LABEL_COLOR_MAP = {idx:val for idx, val in enumerate(color_list[:cluster_count])}
label_color = [LABEL_COLOR_MAP[l] for l in X_pred]

# Plot the scatter digram
plt.figure(figsize = (7,7))
plt.scatter(pca_d[:,0],pca_d[:,1], c= label_color, alpha=0.5) 
plt.show()

