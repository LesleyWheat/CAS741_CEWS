# File description
# Constructs a relative neighbourhood graph over a dataset
#----------------------------------------------------------------
# Import libraries
import numpy as np
import scipy.spatial
from math import isclose

# Local constants
TOL_FLOAT = 1e-10

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
    distance_ab = distanceMatrix[a, b]

    if (a == b):
        # Do not create edges when points are the same
        E = False
    if isclose(distance_ab, 0, abs_tol=TOL_FLOAT):
        # Do not create edges when points are on top of each other other
        E = False
    else:
        E = True
        for i in range(0, n):
            # Only check points not on top of each other
            if (distanceMatrix[a,i] > 0) & (distanceMatrix[b,i] > 0):
                max_dist = max(distanceMatrix[a,i], distanceMatrix[b,i])

                if not isclose(max_dist, distance_ab, abs_tol=TOL_FLOAT) and max_dist < distance_ab:
                    # if too close to call, then consider making edge
                    E = False
                    break

    return(E)
