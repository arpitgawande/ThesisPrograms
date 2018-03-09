
# coding: utf-8

# In[ ]:

# Data Normalization/Scalling
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

# Calculating Eigenvectors and eigenvalues of Cov matirx
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


# In[ ]:

#Cluster Beerstyle and Reviewrs
from sklearn.cluster import KMeans
beerStylesCluster =  KMeans(n_clusters=5).fit_predict(beerStyles)
#Use PCA component analysis for visuals
from sklearn.decomposition import PCA
beerStyles_reduced_data = PCA(n_components=5).fit_transform(beerStyles)

LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b', 3 : 'm', 4 : 'y'}
label_color = [LABEL_COLOR_MAP[l] for l in beerStylesCluster]
# Plot the scatter digram
plt.figure(figsize = (7,7))
plt.scatter(beerStyles_reduced_data[:,0],beerStyles_reduced_data[:,1], c= label_color, alpha=0.5) 
plt.show()

