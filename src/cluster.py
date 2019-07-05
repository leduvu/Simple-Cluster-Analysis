################################################################################
# Python 2.7
# Tested on MacOS Sierra
# 
# Cluster analysis
# A simple program which displays the elbow method and k-means (not seperated).
#
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.scatter.html
################################################################################

from sklearn.cluster  import KMeans # K-means algorithm
import pylab                        # Plotting graphics
import pandas                       # Reading csv data

# Initialize file and cluster number after seeing the result of the elbow method
# or just change to see what happens
filename = 'data-set1.csv'
n_clusters = 5

# READING DATA
data = pandas.read_csv(filename)
column1 = data[[data.columns[0]]]
column2 = data[[data.columns[1]]]

#----------------------------------------------------------------------------
# ELBOW METHOD (Clustering)
# Finding the most likely number of cluster in the dataset
cluster_range = range(1, 10)		# Range of possible clusters

# Calculate the % of variance for each number of clusters

# Initialize with the whole cluster range and calculate for all ranges the k-means
kmeans = [KMeans(n_clusters=i) for i in cluster_range]
score = [kmeans[i].fit(data).score(data) for i in range(len(kmeans))]

# Display it
pylab.plot(cluster_range,score)           # plot(x,y)
pylab.xlabel('Number of Clusters')        # label at the x axis
pylab.ylabel('Score')                     # label at the y axis
pylab.title('Elbow Curve')

pylab.show()

#----------------------------------------------------------------------------

# DISPLAY THE RESULT WITH THE CALCULATED 'RIGHT' CLUSTER NUMBER
# K-means with chosen cluster number
kmeans = KMeans(n_clusters, max_iter=4000)  # Initialize
kmeansoutput = kmeans.fit(data)             # Compute k-means clustering for the whole data

pylab.figure('Cluster K-Means')
pylab.scatter(column1, column2, c=kmeansoutput.labels_) # scatter(x,y, colors)

pylab.show()
