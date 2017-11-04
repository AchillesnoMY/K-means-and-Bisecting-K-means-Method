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

#Implementation of bisecting K-means method
def biKmeans(dataSet,K,numIterations):
    m=shape(dataSet)[0]
    centroid0=mean(dataSet,axis=0).tolist()[0]
    centList=[centroid0]
    clusterAssment=mat(zeros((m,2)))
    
    #store the SSE for the intial cluster0 which is the whole dataset
    for i in range(m):
        clusterAssment[i,1]=distEclud(mat(dataSet[i,:]), mat(centroid0))**2
    while len(centList)<K:
        #Choose the cluster with the largest SSE from cluster list to split two smaller
        #clusters
        SSE1=-1.0; maxIndex=-2
        for j in range(len(centList)):
            SSE=sum(clusterAssment[nonzero(clusterAssment[:,0]==j)[0],1])**2
            if SSE>SSE1:
                maxIndex=j
                SSE1=SSE       
        SSE2=inf;
        #For numIterations times, take the split that produces the clustering 
        #with smallest total SSE (highest overall similarity)
        for m in range(numIterations):
            ptsInCurrCluster=dataSet[nonzero(clusterAssment[:,0]==maxIndex)[0],:]
            centroidMat,splittedClusterAssment=kMeans(ptsInCurrCluster, 2)
            totalSSE=sum(splittedClusterAssment[:,1],axis=0)**2
            if totalSSE<SSE2:
                tempCentr=centroidMat.copy()
                tempClusterAssment=splittedClusterAssment.copy()
                SSE2=totalSSE
        tempClusterAssment[nonzero(tempClusterAssment[:,0]==1)[0],0]=len(centList)
        tempClusterAssment[nonzero(tempClusterAssment[:,0]==0)[0],0]=maxIndex
    
        #update the error in clusterAssment
        numOfDataInCluster=len(tempClusterAssment[:,0])
        clusterAssment[nonzero(clusterAssment[:,0]==maxIndex)[0],1]\
        =tempClusterAssment[:,1].reshape(numOfDataInCluster).tolist()[0] #Handle boardcasting
    
        #update the label index of data
        length=len(tempClusterAssment)
        clusterAssment[nonzero(clusterAssment[:,0]==maxIndex)[0],0]=tempClusterAssment[:,0].reshape(length).tolist()[0]
        
        #Add two choosen centroids to centList
        centList[maxIndex]=centroidMat[0,:]
        centList.append(centroidMat[1,:])
    
    return centList,clusterAssment
        


     
    