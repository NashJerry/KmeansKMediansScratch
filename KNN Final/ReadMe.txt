The random seed line (line 80) looks as folows:
	np.random.seed(X)
The change the seed number, alter the values of x with another interger. To have random inital centroids chosen
each time, comment out this line.

By default the algorithm produces two graphs, one with a K-means value, and one with a K-medians value. 
To run K-means and K-Medians on unnormalised data, uncomment line 267 and 270. to run K-means and K-medians on 
normalised data, uncomment line 268 and line 271. 