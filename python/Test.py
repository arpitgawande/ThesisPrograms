
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


d = dict()
d ={i: i**2 for i in range(10)}


# In[4]:


df = pd.DataFrame.from_dict(d, orient="index")


# In[5]:


df.head()


# In[35]:


nums = [1, 2, 3, 4, 5]  


# In[50]:


nums[-1]


# In[22]:


(i for i in nums)


# In[9]:


list(df.columns)


# In[15]:


it = iter(nums)


# In[20]:


next(it)


# In[52]:


a = 1


# In[53]:


'a' + str(a+2)


# In[9]:


d = {'A': 1, 'B': 2, 'C': 3, '':1}
print(d)
{key:value for key, value in d.items() if key != ''}


# In[11]:


tdf = pd.read_csv('converted/s1', index_col=0)
#Filter Columns
t = tdf[['ip.dst', 'ip.proto', 'sniff_timestamp', 'sample']]
#Remove null destinations
t = t[t['ip.dst'].notnull()]
#Rename Columns
t.columns = ['ip', 'proto', 'time_stamp', 'sample']
#Get count for each ip
df = t.groupby(['ip', 'proto']).size().unstack().fillna(0).astype(int)


# In[15]:


df.head()


# In[17]:


df[[6,17]]

