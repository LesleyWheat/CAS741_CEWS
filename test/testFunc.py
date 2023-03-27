# File description

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

#----------------------------------------------------------------
# Functions

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

def getN1(data, labels):
    return(MFEComplexity.ft_n1(data, labels))