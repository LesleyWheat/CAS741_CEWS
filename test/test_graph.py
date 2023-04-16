# Description

# Imports

# Core
import sys
import os

# Packages
import pytest
import numpy as np

# Project files
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
import graphModule

# --------------------------------------------------------------------
# Functions to setup tests

def TC_square():
    data = [[0, 0], [1, 0], [0, 1], [1, 1]]
    expectedMatrix = [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]
    data = np.asarray(data, dtype=np.float64)
    expectedMatrix = np.asarray(expectedMatrix, dtype=np.int16)
    return (data, expectedMatrix)

def TC_line():
    data = [[0, 0], [1, 1], [2, 2], [3, 3]]
    expectedMatrix = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    data = np.asarray(data, dtype=np.float64)
    expectedMatrix = np.asarray(expectedMatrix, dtype=np.int16)
    return (data, expectedMatrix)

# Function for tests
def assertAllEdges(graphExpected, graphResult):
    n = graphExpected.shape[0]

    for i in range(0, n):
        for j in range(0, n):
            edge_expected = graphExpected[i, j]
            edge_result = graphResult[i,j]

            assert edge_expected == edge_result

# --------------------------------------------------------------------
# Tests


@pytest.mark.FP
@pytest.mark.GM
@pytest.mark.VnV
def test_GM_square():
    data, expectedMatrix = TC_square()
    graph = graphModule.relativeNeighbourGraph(data)
    assertAllEdges(expectedMatrix, graph)

@pytest.mark.FP
@pytest.mark.GM
@pytest.mark.VnV
def test_GM_line():
    data, expectedMatrix = TC_line()
    graph = graphModule.relativeNeighbourGraph(data)
    assertAllEdges(expectedMatrix, graph)