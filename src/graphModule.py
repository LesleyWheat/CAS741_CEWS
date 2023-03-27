# File description
# Constructs a relative neighbourhood graph over a dataset
#----------------------------------------------------------------
# Import libraries
import numpy as np
import scipy.spatial

#----------------------------------------------------------------
# Global Functions
def relativeNeighbourGraph(data):
    # Constructs the relative neighbour graph for a given 
    # set of points

    # Calculate distance between all points
    distanceMatrix = scipy.spatial.distance_matrix(data, data)

    # Iterate through all pairs of points to find edges
    n = distanceMatrix.shape[0]
    V = np.zeros(distanceMatrix.shape)
    for i in range(0, n-1):
        for j in range(i+1, n):
            edge = checkEdge_RNG(distanceMatrix, i, j)
            V[i,j] = edge
            V[j,i] = edge

    return(V)

#----------------------------------------------------------------
# Local Functions
def checkEdge_RNG(distanceMatrix, a, b):
    # Check if an edge exists between two points
    n = distanceMatrix.shape[0]
    E = True

    if (a == b):
        E = False
    else:
        distance_ab = distanceMatrix[a, b]
        for i in range(0, n):
            if (not (i == a or i == b)):
                if max(distanceMatrix[a,i], distanceMatrix[b,i]) < distance_ab:
                    E = False
                    break

    return(E)
