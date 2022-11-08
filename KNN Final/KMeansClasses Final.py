import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

animals = (pd.read_csv('animals', sep=' ', header=None)).to_numpy()
countries = (pd.read_csv('countries', sep=' ', header=None)).to_numpy()
fruits = (pd.read_csv('fruits', sep=' ', header=None)).to_numpy()
veggies = (pd.read_csv('veggies', sep=' ', header=None)).to_numpy()

allData = np.concatenate((animals, countries, fruits, veggies))
noTextAllData = allData[:, 1:]
byLength = [len(animals), len(countries), len(fruits), len(veggies)]


def DataNormalised():
    '''
    returns the normalised data
    '''
    normalisedData = []
    for rowEntry in noTextAllData:
        # each row is replaced with its normalised and appended to an empty list
        RowNormed = np.insert(
            ((rowEntry)/np.linalg.norm(rowEntry)), 0, rowEntry)
        normalisedData.append(RowNormed)
    return np.array(normalisedData)


# lists to be used when plotting kmeans
PrecisionForPlotting = []
RecallForPlotting = []
fScoreForPlotting = []

# lists to be used when plotting kmedians
MediansPrecisionForPlotting = []
MediansRecallForPlotting = []
MediansfScoreForPlotting = []

# to get a plot from k 1-9
for k in range(1, 10):
    def euclidean_Dist(row, centroid):
        '''
        calculates the euclidean distance
        Parametres:
            Row: the current row being iterated over
            Centroid : measurement fromt the current centroid
        Returns:
        the euclidean distance
        '''
        return np.linalg.norm(row - centroid)

    def manhattanDistance(row, centroid):
        # similar to euclidean distance but return manhattan distance instead
        return np.sqrt(np.linalg.norm(row - centroid))

    class KMeans:
        '''
        This is the superclass to which KMedians will also belong
        '''

        def __init__(self, Kclusters=k, maxIterations=100):
            self.k = Kclusters
            self.maxIterations = maxIterations
            # a list of sample indicies for each cluster
            self.clusters = [[] for i in range(self.k)]
            # centroids belonging to the indicies above
            self.centroids = []

        def predictData(self, dataSet):
            '''
            main function that will run the required other functions to obtain kMeans/kMedians
            Parametres:
                input - the dataset to be assessed
            Returns:
                the converged centroids and clusters
            '''
            self.dataSet = dataSet
            self.numOfSample, self.numOfFeatures = dataSet.shape
            # randomly picked rows from data are set to be the centroids, first by index then the rows at that index
            np.random.seed(10)
            InitialCentIndexes = np.random.choice(self.numOfSample, self.k)
            self.centroids = [self.dataSet[index]
                              for index in InitialCentIndexes]

            # optimisation
            counter = 0
            for i in range(self.maxIterations):
                # update clusters
                self.clusters = self.makeClusters(self.centroids)
                # the old centroids are updated and new centroids are created through the get centroids function
                OldCentroids = self.centroids
                self.centroids = self.getCentroids(self.clusters)
                # check convergance function assesses whether the data has converged
                if self.checkCovergance(OldCentroids, self.centroids) != True:
                    counter += 1
                if self.checkCovergance(OldCentroids, self.centroids) == True:
                    counter += 1
                    finalClusters = self.clusters
                    self.precisionCalculator(finalClusters)
                    #print("It took", counter, "iterations to converge.")
                    break
            return finalClusters, OldCentroids

        def makeClusters(self, centroids):
            '''
            assign samples to closest centroids to make out clusters

            Parameters:
                centroids : the current centroids in the dataset
            Returns:
                the clusters around those centroids
            '''
            # intial empty clusters
            clusters = [[] for i in range(self.k)]
            for Currentindex, CurrentSample in enumerate(self.dataSet):
                # calls function which finds the centroid in closest proximity
                centroid_index = self.proximalCentroid(
                    CurrentSample, centroids)
                clusters[centroid_index].append(Currentindex)
            return clusters

        def proximalCentroid(self, CurrentSample, centroids):
            '''
            finds the centroid in closest proximity to the current row.

            Parameters:
                CurrentSample :  the row being checked against in the dataset
                centroid : a list of all the currently defined centoids
            Returns:
                the index of the closest centroid
            '''
            # calculate the distance of the current sample row from all the centroids
            pointDistances = [euclidean_Dist(
                CurrentSample, point) for point in centroids]
            # select the index is the closes
            indexOfClosest = np.argmin(pointDistances)
            return indexOfClosest

        def getCentroids(self, clusters):
            '''
            creates new centroids according to the mean of the clusters
            Parameter:
                cluster : the cluster
            Returns :
                the new centroid
            '''
            # this trigger means that it is kMeans rather than median being run
            self.trigger = 1
            centroids = np.zeros((self.k, self.numOfFeatures))
            for clusterIndex, Cluster in enumerate(clusters):
                if len(Cluster) != 0:
                    # mean is calculated for the new centroid
                    clusterMean = np.mean(self.dataSet[Cluster], axis=0)
                    centroids[clusterIndex] = clusterMean
                # if the cluster is empty the mean remains as is
                else:
                    pass
            return centroids

        def checkCovergance(self, OldCentroids, centroids):
            '''
            Assesses if the centroids have converged, i.e the clusters remain the same.

            Parameters:
                old Centroids 
                cebtroids : the current Centroids
            Returns:
                boolean value on whether they have converged or not
            '''
            distances = [euclidean_Dist(
                OldCentroids[i], centroids[i]) for i in range(self.k)]
            # if the distance is 0 it means the old centroid and new one is exactly the same so they have converged
            if sum(distances) == 0:
                return True
            if sum(distances) != 0:
                return False

        def precisionCalculator(self, ultimateClusters):
            '''
            Calculates the BCubed measures of the data.

            Parameter:
                ultimateClusters : the final clusters after the data has converged
            Returns:
                Updated lists containing the precision, recall and Fscores. appended outside the function.
            '''
            animalsCounted, countriesCounted, fruitsCounted, veggiesCounted = [], [], [], []
            completelist = []
            for i in ultimateClusters:
                # each cluster's value for each category is counted depending on their index
                animalCounter, countryCounter, fruitsCounter, veggiesCounter = 0, 0, 0, 0
                for j in i:
                    if 0 <= j < 50:
                        animalCounter += 1
                    if 50 <= j < 211:
                        countryCounter += 1
                    if 211 <= j < 269:
                        fruitsCounter += 1
                    if 269 <= j < 327:
                        veggiesCounter += 1
                animalsCounted.append(animalCounter)
                countriesCounted.append(countryCounter)
                fruitsCounted.append(fruitsCounter)
                veggiesCounted.append(veggiesCounter)
                newBigList = [animalCounter, countryCounter,
                              fruitsCounter, veggiesCounter]
                completelist.append(newBigList)
            allRecalls = []
            allPrecisions = []
            allFScores = []
            for Finalcluster in completelist:
                clusterLabel = (max(Finalcluster))
                sumI = sum(Finalcluster)
                for element in Finalcluster:
                    precision = ((element/sumI)*element)*100  # precision is calculated
                    allPrecisions.append(precision)
                    # recall is calculated
                    individualRecall = ((element/byLength[Finalcluster.index(element)])*element) * 100
                    allRecalls.append(individualRecall)
                    # fScore calculated
                    if precision == 0 or individualRecall == 0:
                        individualFScore = 0
                    else:
                        individualFScore = (2*precision*individualRecall)/(precision+individualRecall)
                    allFScores.append(individualFScore)
            averagedRecall = sum(allRecalls)/327
            finalPrecisions = sum(allPrecisions)/327
            finalFScores = sum(allFScores)/327
            print("the precision at k = ", k, "is", finalPrecisions)
            print("the recall at k = ", k, "is", averagedRecall)
            print("the F-Score at k = ", k, "is", finalFScores)
            print("\n")
            # the trigger determines to which list outside the function, the measures are added
            if self.trigger == 1:
                PrecisionForPlotting.append(finalPrecisions)
                RecallForPlotting.append(averagedRecall)
                fScoreForPlotting.append(finalFScores)
            if self.trigger == 2:
                MediansPrecisionForPlotting.append(finalPrecisions)
                MediansRecallForPlotting.append(averagedRecall)
                MediansfScoreForPlotting.append(finalFScores)

    class kMedian(KMeans):
        def getCentroids(self, clusters):
            # having the trigger be 1 rather than 2 means the kmedians is tun instead
            self.trigger = 2
            centroids = np.zeros((self.k, self.numOfFeatures))
            for clusterIndex, Cluster in enumerate(clusters):
                if len(Cluster) != 0:
                    clusterMean = np.median(self.dataSet[Cluster], axis=0)
                    centroids[clusterIndex] = clusterMean
                else:
                    pass
            return centroids

        def checkCovergance(self, OldCentroids, centroids):
            distances = [manhattanDistance(
                OldCentroids[i], centroids[i]) for i in range(self.k)]
            if sum(distances) == 0:
                return True
            if sum(distances) != 0:
                return False

    x1 = KMeans()
    #x1.predictData(noTextAllData)
    normalisedData = DataNormalised()
    x1.predictData(normalisedData)

    x2 = kMedian()
    #x2.predictData(noTextAllData)
    normalisedData = DataNormalised()
    x2.predictData(normalisedData)


def plotting():
    fig, axis = plt.subplots(2, 1)
    axis[0].plot(PrecisionForPlotting, marker='o', label='Precision')
    axis[0].plot(RecallForPlotting, marker='s', label='Recall',)
    axis[0].plot(fScoreForPlotting, marker='*', label='FScore')
    axis[0].set_title("K-Means Precision, Recall & F-Score At K 1-9")
    axis[0].legend()

    axis[1].plot(MediansPrecisionForPlotting, marker='o', label='Precision')
    axis[1].plot(MediansRecallForPlotting, marker='s', label='Recall',)
    axis[1].plot(MediansfScoreForPlotting, marker='*', label='FScore')
    axis[1].set_title("K-Medians Precision, Recall & F-Score At K 1-9")
    axis[1].legend()

    plt.xlabel("K-1")
    plt.ylabel("Overall Score")
    plt.tight_layout()
    plt.show()

plotting()
