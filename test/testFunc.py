# File description
# Misc functions used for testing.

#----------------------------------------------------------------
# Imports

#Core
import sys
import os

# Packages
import numpy as np
import scipy.spatial
from relativeNeighborhoodGraph import returnRNG
from pymfe.complexity import *

# Import local files
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
import graphModule
import CEWS

#----------------------------------------------------------------
# Functions

# Cound edges made by graph module
def countEdgesFromSrc(data, labels):
    V = graphModule.relativeNeighbourGraph(data)
    n = V.shape[0]

    sumCutEdges = 0
    sumUncutEdges = 0

    for i in range(0, n-1):
        for j in range(i+1, n):
            if(V[i,j]):
                if(labels[i] == labels[j]):
                    sumUncutEdges = sumUncutEdges + 1
                else:
                    sumCutEdges = sumCutEdges + 1

    sumAllEdges = sumCutEdges + sumUncutEdges

    edges = {
      "cut": sumCutEdges,
      "uncut": sumUncutEdges,
      "all": sumAllEdges
    }

    return(edges)

# count edges from graph made by relative neighbourhood library
def countEdgesFromLibrary(data, labels):
    DM = scipy.spatial.distance_matrix(data, data)
    V = returnRNG.returnRNG(DM)
    n = V.shape[0]

    sumCutEdges = 0
    sumUncutEdges = 0

    for i in range(0, n-1):
        for j in range(i+1, n):
            if V[i][j] > 0:
                if(labels[i] == labels[j]):
                    sumUncutEdges = sumUncutEdges + 1
                else:
                    sumCutEdges = sumCutEdges + 1

    sumAllEdges = sumCutEdges + sumUncutEdges

    edges = {
      "cut": sumCutEdges,
      "uncut": sumUncutEdges,
      "all": sumAllEdges
    }


    return(edges)

# Compare each edge between graph module and relativeNeighorhoodGraph package
def compareEdges(data, label):
    V_G = graphModule.relativeNeighbourGraph(data)
    DM = scipy.spatial.distance_matrix(data, data)
    V_L = returnRNG.returnRNG(DM)
    n = DM.shape[0]

    allEdgesMatch = True

    for i in range(0, n-1):
        for j in range(i+1, n):
            edgeG = V_G[i][j]
            edgeL = (V_L[i][j] > 0)
            if edgeG != edgeL:
                print("i: " + str(i) + " j: " + str(j) + " VG: "+ str(V_G[i][j]) + " V_L: " + str(V_L[i][j]) + " distance: " + str(DM[i][j]))
                print("Point i: " + str(data[i]) + " Point j: " + str(data[j]))
                allEdgesMatch = False

    return allEdgesMatch

# Get complexity measure
def getN1(data, labels):
    return(MFEComplexity.ft_n1(data, labels))

# Test calculation by randomizing
def randomizeDataJn(data, labels):
    assert data.shape[0] == labels.shape[0] 

    # Randomize data and labels while keeping association
    rng = np.random.default_rng()
    p = rng.permutation(data.shape[0])
    data = data[p]
    labels = labels[p]

    return CEWS.cutEdgeWeight(data, labels)

def randomizeEdges(data):
    print("G1")
    V_G1 = graphModule.relativeNeighbourGraph(data)

    rng = np.random.default_rng()
    n = data.shape[0]
    p = rng.permutation(n)
    dataRan = data[p]

    print("G2")
    V_G2 = graphModule.relativeNeighbourGraph(dataRan)

    for i in range(0, n-1):
        for j in range(i+1, n):
            rand_i = np.where(p == i)[0][0]
            rand_j = np.where(p == j)[0][0]
            edge1 = V_G1[i, j]
            edge2 = V_G2[rand_i, rand_j]
            if edge1 != edge2:
                print("i: " + str(i) + " j: " + str(j))
                print("ri: " + str(rand_i) + " rj: " + str(rand_j))
                print("pi: " + str(data[i]) + " ri: " + str(dataRan[rand_i]))
                print("pj: " + str(data[j]) + " rj: " + str(dataRan[rand_j]))
                print("1: " + str(edge1) + " 2: " + str(edge2))
                assert edge1 == edge2
