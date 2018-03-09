
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
# Common Imports
import numpy as np
#Pandas for creating dataframes
import pandas as pd
#Sklearn
from sklearn import preprocessing
#K-means clustering algo
from sklearn.cluster import KMeans
#OS moduled for file oprations
import os
#CSV module
import csv
#Plotting
import matplotlib.pyplot as plt


# In[2]:

# Calculating Eigenvectors and eigenvalues of Cov matirx
def PCA_component_analysis(X_std):
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

    # # Sort from high to low
    eig_pairs.sort(key = lambda x: x[0], reverse= True)

    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
    cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

    # PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 
    plt.figure(figsize=(10, 5))
    plt.bar(range(9), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')
    plt.step(range(9), cum_var_exp, where='mid',label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()


# In[5]:

from sklearn.decomposition import PCA
def plot_clusters(X, X_clusters, centroids, kmeans):
    #Use PCA component analysis for visuals
    if X.shape[1] > 2:
        reduced_X = PCA(n_components=2).fit_transform(X)
    else:
        reduced_X = X
   
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .01     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_X[:, 0].min() - 1, reduced_X[:, 0].max() + 1
    y_min, y_max = reduced_X[:, 1].min() - 1, reduced_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')   
    #Plot the data points (PCA reduced components)
    plt.plot(reduced_X[:,0],reduced_X[:,1],  'k.', markersize=3) 
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
    plt.title('K-means clustering with (PCA-reduced data), Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# In[6]:

from scipy.spatial.distance import euclidean
def k_mean_dist(data, clusters, cluster_centers):
    distances = []
    for i, d in enumerate(data):
        center = cluster_centers[clusters[i]]
        distance = euclidean(d,center)
        #distance = np.linalg.norm(d - center)
        distances.append(distance)
    return distances


# In[7]:

from sklearn.cluster import KMeans
def determine_cluster_count(X_trans):
    Nc = range(1, 10)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    kmeans
    score = [kmeans[i].fit(X_trans).score(X_trans) for i in range(len(kmeans))]
    score
    plt.plot(Nc,score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()


# In[8]:

import glob
#Merge sample files to create bigger sameple
def merge_sample_files(sample_folder, file_count):
    file_number = 1
    count = 0
    filenames = sorted(glob.glob(os.path.join(sample_folder,'*')),  key=os.path.getmtime)
    for filename in filenames:
        if count == 0:
            df = pd.read_csv(filename, index_col=0)
            count += 1
        else:
            temp_df = pd.read_csv(filename, index_col=0)
            df = df.append(temp_df)
            count += 1
        if count == file_count:
            df.to_csv(os.path.join(sample_folder,'m'+str(file_number)))
            df = df.drop(df.index, inplace=True)
            count = 0
            file_number +=1


# In[13]:

def get_cluster_feature_vector_dict(cluster_folder):
    cluster_dict = dict()
    filenames = sorted(glob.glob(os.path.join(cluster_folder,'*')),  key=os.path.getmtime)
    first = True
    for filename in filenames:
        if first:
            df = pd.read_csv(filename, index_col=0)
            first = False
        else:
            temp_df = pd.read_csv(filename, index_col=0)
            df = df.append(temp_df) 
    df = df.reset_index().set_index(['cluster','ip'])
    clusters = df.index.get_level_values(0).unique()
    for c in clusters:
        cluster_dict[c] = df.loc[c].iloc[:,:-1].values
    return cluster_dict


# In[72]:

def plot_outlier_detecton(X_train, clf):
    
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    
    # plot the levels lines and the points
    print(clf.name)
    if clf.name == "lof":
        # decision_function is private for LOF
        Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    print(Z.max(), Z.min())
    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    #plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
    
    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1],
           ["learned frontier", "training observations"],
           loc="upper left")
    plt.show()


# In[73]:

#SKlearn SVM
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
def one_class_svm_for_clusters(cluster_feature_dict):
    svm_dict = dict()
    scalar_dict = dict()
    for key, value in cluster_feature_dict.items():
        X_train = value
        #Get scaler
        scaler = preprocessing.StandardScaler().fit(X_train)
        scalar_dict[key] = scaler 
        #Transform Traning data
        X_trans = scaler.transform(X_train)
        # fit the model
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(X_trans)
        clf.name = 'svm'
        plot_outlier_detecton(X_trans, clf)
        
        clf = LocalOutlierFactor(contamination=0.01)
        clf.fit(X_trans)
        clf.name= 'lof'
        plot_outlier_detecton(X_trans, clf)
        #Store trained SVM for each IP
        svm_dict[key] = clf
    return svm_dict, scalar_dict


# In[15]:

#Base Folder sPaths
base_path = os.path.join('converted','test2')
#Normal
sample_path = os.path.join(base_path,'samples')
cluster_path = os.path.join(base_path,'ip_cluster')
#Attack
# sample_path = os.path.join(base_path,'attack_samples','1')
# cluster_path = os.path.join(base_path,'attack_ip_cluster','1')

centroid_path = os.path.join(base_path,'centroids')


# In[59]:

#merge_sample_files(sample_path,3)
d = get_cluster_feature_vector_dict(cluster_path)


# In[77]:

#svm_dict, scaler_dict = one_class_svm_for_clusters(d)


# In[27]:

svm_dict


# In[50]:

#Predict for the give destination if it is normal or not
X_test = [[0,120]]
X_test_tran = scaler_dict[0].transform(X_test)
svm_dict[0].predict(X_test_tran)


# In[32]:

first = True
ip_dict = dict()
sample_count = 1;
columns = ['0','1','2','3']
centroid_dfs = []
first = True

for filename in os.listdir(sample_path):
    tdf = pd.read_csv(sample_path+filename, index_col=0)
    #Filter Columns
    t = tdf[['ip.dst', 'ip.proto', 'sniff_timestamp', 'sample']]
    #Remove null destinations
    t = t[t['ip.dst'].notnull()]
    #Rename Columns
    t.columns = ['ip', 'proto', 'time_stamp', 'sample']
    #Get count for each ip
    df = t.groupby(['ip', 'proto']).size().unstack().fillna(0).astype(int)
    #Select TCP and UDP as only fetures (TCP:6, UDP:17)
    df = df[[6,17]]
    #Get value matrix
    X = df.values
    #Create scaling
    scaler = preprocessing.StandardScaler().fit(X)
    #Transform Traning data
    X_trans = scaler.transform(X)
    #print(X_trans)
    #Determine cluster counts using elbow method
    #determine_cluster_count(X_trans)
    #Define Number of Clusters
    cluster_count = 3
    #Data Fitting using K-means
    #if first:
    kmeans = KMeans(n_clusters=cluster_count)
    kmeans.fit(X_trans)
    #Insert cluster center to its corrosposnding dataframe each dataframe.
    #Dataframe 0 contain all the clusters centers associated with 0th cluster 
    for i in range(kmeans.cluster_centers_.shape[0]):
        s = pd.Series(kmeans.cluster_centers_[i], index=df.columns)
        if(first):
            centroid_dfs.append(pd.DataFrame(columns=df.columns))
        centroid_dfs[i] = centroid_dfs[i].append(s,ignore_index=True)         
    first = False


# In[33]:

#Calculate Centroid Mean
centroids = []
features = set()
for df in centroid_dfs:
    centroid = []
    for c in df.columns:
        df = df[np.abs(df[c] - df[c].mean()) <= (3*df[c].std())]
        #print(df[c])
        centroid.append(df[c].mean())
    centroids.append(centroid)
    features |= set(df.columns)
#Save centroid for future clusterinng
if not os.path.exists(centroid_path):
    os.makedirs(centroid_path)
np.savetxt(centroid_path+"centroids.csv", np.asarray(centroids), delimiter=",")
np.savetxt(centroid_path+"features.csv", np.asarray(list(features)), delimiter=",")


# In[44]:

#Actual Clustering

#Get centroid created in initial step
centroids = np.genfromtxt(centroid_path+"centroids.csv", delimiter=',')
features = np.genfromtxt(centroid_path+"features.csv", delimiter=',')
sample_count = 1
for filename in os.listdir(sample_path):
    tdf = pd.read_csv(sample_path+filename, index_col=0)
    #Filter Columns
    t = tdf[['ip.dst', 'ip.proto', 'sniff_timestamp', 'sample']]
    #Remove null destinations
    t = t[t['ip.dst'].notnull()]
    #Rename Columns
    t.columns = ['ip', 'proto', 'time_stamp', 'sample']
    #Get count for each ip
    df = t.groupby(['ip', 'proto']).size().unstack().fillna(0).astype(int)
    #Select TCP and UDP as only fetures (TCP:6, UDP:17)
    df = df[[6,17]]
    if(set(df.columns) != set(features)):
        print(df.columns, features)
        non_columns = set(features) - set(df.columns)
        for c in non_columns:
            df.insert(loc=1, column=c, value=0)
    #Get value matrix
    X = df.values
    #Create scaling
    scaler = preprocessing.StandardScaler().fit(X)
    #Transform Traning data
    X_trans = scaler.transform(X)
    #Data Fitting using K-means
    kmeans = KMeans(n_clusters=centroids.shape[0], init=centroids)
    clusters = kmeans.fit_predict(X_trans)
    #Plot clusters and data using PCA component analysis
    #plot_clusters(X_trans, clusters, centroids, kmeans)
    distances = k_mean_dist(X_trans, clusters, centroids)
    #Attaching label/cluster to IP
    cluster_df = pd.DataFrame({'cluster': kmeans.labels_})
    #Attaching distance from the cluster for each data point
    distance_df = pd.DataFrame({'distance': distances})
    ip_label_df = pd.concat([df.reset_index(), cluster_df, distance_df], axis=1).set_index('ip')
    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)
    ip_label_df.to_csv(cluster_path+str(sample_count))
    sample_count += 1


# In[95]:

#For a given IP address find how many time a given cluster it was assigned to.
from itertools import groupby
def get_IP_cluster_count_dict(cluster_path):    
    ip_dict = dict()
    filenames = glob.glob(os.path.join(cluster_path,'*'))
    for filename in filenames:
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ip'] in ip_dict:
                    #print(row['ip'],ip_dict[row['ip']])
                    ip_dict[row['ip']] = ip_dict[row['ip']] + [row['cluster']]
                else:
                    ip_dict[row['ip']] = [row['cluster']]
    #Find how many time IP was assigned to a given cluster
    ip_cluster_dict = dict()
    for key, value in ip_dict.items():
        ip_cluster_dict[key] = {k: len(list(group)) for k, group in groupby(value)}
    return ip_cluster_dict


# In[96]:

ip_cluster_dict = get_IP_cluster_count_dict(cluster_path)


# In[97]:

ip_cluster_dict


# In[157]:

df = pd.DataFrame([np.arange(5)]*3)


# In[161]:

df[0] = df[0]+1


# In[162]:

df


# In[153]:

[np.arange(10)]*3


# In[ ]:



