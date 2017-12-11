from numpy import *

#load the filename of a txt data file 
#and output a list contains all the data
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine=list(map(float,curLine))
        dataMat.append(fltLine)
    
    return dataMat

#Calculate the Euclidean distance between any two vectors
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#Create a set of random K centroids for each feature of the given dataset
def randCent(dataSet,K):
    n=shape(dataSet)[1]
    centroids=mat(zeros((K,n)))
    for j in range(n):
        minValue=min(dataSet[:,j])
        maxValue=max(dataSet[:,j])
        rangeValues=float(maxValue-minValue)
        #Make sure centroids stay within the range of data
        centroids[:,j]=minValue+rangeValues*random.rand(K,1)
    return centroids

#Implementation of K means clustering method
#The last two parameters in the function can be omitted.
#Output the matrix of all centroids and a matrix (clusterAssment) whose first column represents the 
#belongings of clusters of each obvservation and second column represents the SSE for each
#observation
def kMeans(dataSet,K,distMethods=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,K)
    clusterChanged=True
    
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf; minIndex=-2
            for j in range(K):
                distJI=distMethods(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            if clusterAssment[i,0] != minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        #update all the centroids by taking the mean value of relevant data
        for cent in range(K):
            ptsInClust=dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment           

#plot the K-means of 3 dimensionality (3 features) and project into 2D 
#Use PCA method to reduce the dimensionality from 3 to 2
#The K is 4 for this case
def plotXDKmeans(dataSet,clusterAssment,centroids,K):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    pca=PCA(n_components=2).fit(dataSet)
    pcs_2d=pca.transform(dataSet)
    indexList=clusterAssment[:,0]
    m=shape(indexList)[0]
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(m):
        if indexList[i]==0:
            c1=ax.scatter(pcs_2d[i,0],pcs_2d[i,1],c='r',marker='+')
        elif indexList[i]==1:
            c2=ax.scatter(pcs_2d[i,0],pcs_2d[i,1],c='g',marker='o')
        elif indexList[i]==2:
            c3=ax.scatter(pcs_2d[i,0],pcs_2d[i,1],c='b',marker='*')
        else:
            c4=ax.scatter(pcs_2d[i,0],pcs_2d[i,1],c='y',marker='^')
    ax.legend([c1,c2,c3,c4],['cluster 0','cluster 1','cluster 2','cluster 3'])
            
    plt.show()

#Plot the k-means of 2 dimensionality
#The K is 4 for this case 
def plot2DKmeans(dataSet,label,centroids,K):
    import matplotlib.pyplot as plt
    
    m=shape(dataSet)[0]
    indexList=label
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    for i in range(m):
        if indexList[i]==0:
            c1=ax.scatter(dataSet[i,0],dataSet[i,1],c='r',marker='+')
        elif indexList[i]==1:
            c2=ax.scatter(dataSet[i,0],dataSet[i,1],c='g',marker='o')
        elif indexList[i]==2:
            c3=ax.scatter(dataSet[i,0],dataSet[i,1],c='b',marker='*')
        else:
            c4=ax.scatter(dataSet[i,0],dataSet[i,1],c='y',marker='^')
    ax.legend([c1,c2,c3],['cluster 0','cluster 1','cluster 2','cluster 3'])
    plt.show()       

#bisecting K-means method
def bisectingKMeans(dataSet,K,numIterations):
    m,n=shape(dataSet)
    clusterInformation=mat(zeros((m,2)))
    centroidList=[]
    minSSE=inf
    
    #At the first place, regard the whole dataset as a cluster and find the best clusters
    for i in range(numIterations):
        centroid,clusterAssment=kMeans(dataSet, 2)
        SSE=sum(clusterAssment,axis=0)[0,1]
        if SSE<minSSE:
            minSSE=SSE
            tempCentroid=centroid
            tempCluster=clusterAssment
    centroidList.append(tempCentroid[0].tolist()[0])
    centroidList.append(tempCentroid[1].tolist()[0])
    clusterInformation=tempCluster
    minSSE=inf 
    
    while len(centroidList)<K:
        maxIndex=-2
        maxSSE=-1
        #Choose the cluster with Maximum SSE to split
        for j in range(len(centroidList)):
            SSE=sum(clusterInformation[nonzero(clusterInformation[:,0]==j)[0]])
            if SSE>maxSSE:
                maxIndex=j
                maxSSE=SSE
                
        minIndex=-2
        #Choose the clusters with minimum total SSE to store into the centroidList
        for k in range(numIterations):
            pointsInCluster=dataSet[nonzero(clusterInformation[:,0]==maxIndex)[0]]
            centroid,clusterAssment=kMeans(pointsInCluster, 2)
            SSE=sum(clusterAssment[:,1],axis=0)
            if SSE<minSSE:
                minSSE=SSE
                tempCentroid=centroid.copy()
                tempCluster=clusterAssment.copy()
        #Update the index
        tempCluster[nonzero(tempCluster[:,0]==1)[0],0]=len(centroidList)
        tempCluster[nonzero(tempCluster[:,0]==0)[0],0]=maxIndex
        
        #update the information of index and SSE
        clusterInformation[nonzero(clusterInformation[:,0]==maxIndex)[0],:]=tempCluster
        #update the centrolist
        centroidList[maxIndex]=tempCentroid[0].tolist()[0]
        centroidList.append(tempCentroid[1].tolist()[0])
    return centroidList,clusterInformation
        


     
    
