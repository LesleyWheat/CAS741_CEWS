# File description:
# This modules calculates the value of the normalized unweighted
# cut edge weight statistic for a dataset
#----------------------------------------------------------------
# Import libraries
import numpy as np

# Import local files
import graphModule

#----------------------------------------------------------------
# Global functions

def calc(data, labels):
    # Make graph
    V = graphModule.relativeNeighbourGraph(data)

    # Count up edges
    sumCutEdges = 0
    sumUncutEdges = 0

    n = V.shape[0]
    for i in range(0, n-1):
        for j in range(i+1, n):
            if(V[i,j]):
                if(labels[i] == labels[j]):
                    sumUncutEdges = sumUncutEdges + 1
                else:
                    sumCutEdges = sumCutEdges + 1

    # Compute normalized CEWS
    Jn = sumCutEdges/(sumCutEdges + sumUncutEdges)
    return(Jn)