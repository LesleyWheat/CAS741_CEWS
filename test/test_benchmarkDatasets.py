# Description
# This file contains the tests on benchmark datasets from the
# UCI Machine Learning Repository
#--------------------------------------------------------------------
# Imports:
# Core
import sys
import os

# Packages
import pytest
import numpy as np

# Project files
# Import local files
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
import inputModule
from CEWS import *

# Test files
import getDataset
import testFunc

#--------------------------------------------------------------------
# Constants
TOL_EDGE = 1
TOL_JN = 0.1
TOL_N1 = 0.5

#--------------------------------------------------------------------
# Classes
class runBenchmarkDataset:
    # This class contains the data about a benchmark set so the
    # calculations do not have to be repeated for multiple tests.
    # This significantly speeds up computation time.
    # It also contains specific test functions as the same tests are
    # run on the datasets and this means code is not duplicated.

    # Class variables
    data = None
    labels = None
    Jn = None

    edgesCut = None
    edgesUncut = None
    edgesAll = None

    edgesCut_lib = None
    edgesUncut_lib = None
    edgesAll_lib = None

    paper_n = None
    paper_m = None
    paper_k = None
    paper_J = None
    paper_E = None

    # Class functions
    def __init__(self, datasetName):
        # Get the data for the benchmark datasets and
        # set their comparision values from "A statistical approach to class separability"

        if (datasetName == "iris"):
            inData, inLabels = getDataset.dataset_iris()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 150
            self.paper_m = 4
            self.paper_k = 3
            self.paper_J = 0.087
            self.paper_E = 196
        elif (datasetName == "irisBezdek"):
            inData, inLabels = getDataset.dataset_irisBezdek()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 150
            self.paper_m = 4
            self.paper_k = 3
            self.paper_J = 0.09
            self.paper_E = 189
        elif (datasetName == "breastCancer"):
            inData, inLabels = getDataset.dataset_breastCancer()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 683
            self.paper_m = 9
            self.paper_k = 2
            self.paper_J = 0.008
            self.paper_E = 7562

        elif (datasetName == "wine"):
            inData, inLabels = getDataset.dataset_wine()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 178
            self.paper_m = 13
            self.paper_k = 3
            self.paper_J = 0.093
            self.paper_E = 281

        elif (datasetName == "yeast"):
            inData, inLabels = getDataset.dataset_yeast()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 1484
            self.paper_m = 8
            self.paper_k = 10
            self.paper_J = 0.524
            self.paper_E = 2805

        elif (datasetName == "ionosphere"):
            inData, inLabels = getDataset.dataset_ionosphere()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 351
            self.paper_m = 34
            self.paper_k = 2
            self.paper_J = 0.137
            self.paper_E = 402

        elif (datasetName == "glass"):
            inData, inLabels = getDataset.dataset_glass()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 214
            self.paper_m = 9
            self.paper_k = 6
            self.paper_J = 0.356
            self.paper_E = 275
        elif (datasetName == "haberman"):
            inData, inLabels = getDataset.dataset_haberman()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 306
            self.paper_m = 3
            self.paper_k = 2
            self.paper_J = 0.331
            self.paper_E = 517
        elif (datasetName == "imgSeg"):
            inData, inLabels = getDataset.dataset_imgSeg()
            self.data, self.labels = inputModule.proccessInput(inData, inLabels)

            # Values from paper
            self.paper_n = 210
            self.paper_m = 19
            self.paper_k = 7
            self.paper_J = 0.224
            self.paper_E = 268
        else:
            raise ValueError("Invalid dataset name")
        
        
    def getEdges(self):
        edges = testFunc.countEdgesFromSrc(self.data, self.labels)

        self.edgesCut = edges["cut"]
        self.edgesUncut = edges["uncut"]
        self.edgesAll = edges["all"]

    def getEdgesLibrary(self):
        edges = testFunc.countEdgesFromLibrary(self.data, self.labels)

        self.edgesCut_lib = edges["cut"]
        self.edgesUncut_lib = edges["uncut"]
        self.edgesAll_lib = edges["all"]

    def checkEdgesGot(self):
        if(self.edgesCut == None):
            self.getEdges()

    def checkEdgesLibGot(self):
        if(self.edgesCut_lib == None):
            self.getEdgesLibrary()

    def checkJnGot(self):
        if(self.Jn == None):
            self.Jn = cutEdgeWeight(self.data, self.labels)
    
    #Test functions
    def checkEdgesLibraryAll(self):
        self.checkEdgesGot()
        self.checkEdgesLibGot()
    
        assert self.edgesAll <= self.edgesAll_lib + TOL_EDGE
        assert self.edgesAll >= self.edgesAll_lib - TOL_EDGE

    def checkEdgesLibraryCut(self):
        self.checkEdgesGot()
        self.checkEdgesLibGot()

        assert self.edgesCut <= self.edgesCut_lib + TOL_EDGE
        assert self.edgesCut >= self.edgesCut_lib - TOL_EDGE
        print(self.edgesCut)
        print(self.edgesCut_lib)

    def checkEdgesLibraryUncut(self):
        self.checkEdgesGot()
        self.checkEdgesLibGot()

        assert self.edgesUncut <= self.edgesUncut_lib + TOL_EDGE
        assert self.edgesUncut >= self.edgesUncut_lib - TOL_EDGE
        print(self.edgesUncut)
        print(self.edgesUncut_lib)

    def checkDataset(self):
        assert self.paper_n == self.data.shape[0]
        assert self.paper_m == self.data.shape[1]

        uniqueClasses = np.unique(self.labels)
        k = uniqueClasses.shape[0]
        assert self.paper_k == k
    
    def checkEdgesAll(self):
        self.checkEdgesGot()
    
        assert self.edgesAll <= self.paper_E + TOL_EDGE
        assert self.edgesAll >= self.paper_E - TOL_EDGE
    
    def checkEdgesCut(self):
        self.checkEdgesGot()

        pec = round(self.paper_E*self.paper_J)
        tol_base = round(self.paper_E*0.001)
        assert self.edgesCut <= pec + TOL_EDGE + tol_base
        assert self.edgesCut >= pec - TOL_EDGE - tol_base
        print(self.edgesCut)
        print(pec)

    def checkEdgesUncut(self):
        self.checkEdgesGot()
            
        peuc = round(self.paper_E - self.paper_E*self.paper_J)
        tol_base = round(self.paper_E*0.001)
        assert self.edgesUncut <= peuc + TOL_EDGE + tol_base
        assert self.edgesUncut >= peuc - TOL_EDGE - tol_base
        print(self.edgesUncut)
        print(peuc)
    
    def checkJn(self):
        self.checkJnGot()

        assert self.Jn <= self.paper_J + TOL_JN
        assert self.Jn >= self.paper_J - TOL_JN

    def checkJn_lib(self):
        self.checkEdgesLibGot()
        self.checkJnGot()

        JN_lib = self.edgesCut_lib/self.edgesAll_lib
        assert self.Jn <= JN_lib + TOL_JN
        assert self.Jn >= JN_lib - TOL_JN
    
    def checkN1(self):
        self.checkJnGot()
    
        N1 = testFunc.getN1(self.data, self.labels)
        assert self.Jn <= N1 + max(TOL_N1*N1, 0.05)
        assert self.Jn >= N1 - max(TOL_N1*N1, 0.05)
    
    def checkRandomize(self):
        self.checkJnGot()

        for i in range(1, 10):
            random_Jn = testFunc.randomizeDataJn(self.data, self.labels)
            assert self.Jn == random_Jn


#-----------------------------------------------------------------
# Make test objects
pytest.testObjects_names = ["iris", "irisBezdek", "yeast",
                            "breastCancer","wine","glass",
                            "ionosphere","haberman","imgSeg"]

pytest.benchmarks = list()
for name in pytest.testObjects_names:
    pytest.benchmarks.append([runBenchmarkDataset(name),name])

#-----------------------------------------------------------------
# Tests

#### Check dataset shape/size/classes vs paper Tests ####
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.IM
@pytest.mark.paperCompare
def test_checkDataset(test_obj, name):
    test_obj.checkDataset()

#### Graph vs Paper Tests ####
# Check all edges
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.GM
@pytest.mark.paperCompare
def test_checkEdgesAll(test_obj, name):
    test_obj.checkEdgesAll()

# Check cut edges
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.GM
@pytest.mark.Cut
@pytest.mark.paperCompare
def test_checkEdgesCut(test_obj, name):
    test_obj.checkEdgesCut()

# Check uncut edges
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.GM
@pytest.mark.Uncut
@pytest.mark.paperCompare
def test_checkEdgesUncut(test_obj, name):
    test_obj.checkEdgesUncut()

#### Graph vs RNG Library Tests ####
# Check the graph constructor vs
# a relative neighbourhood graph library
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.graphLibrary
@pytest.mark.GM
def test_checkEdgesLibraryAll(test_obj, name):
    test_obj.checkEdgesLibraryAll()

@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.graphLibrary
@pytest.mark.GM
@pytest.mark.Cut
def test_checkEdgesLibraryCut(test_obj, name):
    test_obj.checkEdgesLibraryCut()

@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.graphLibrary
@pytest.mark.Uncut
@pytest.mark.GM
def test_checkEdgesLibraryUncut(test_obj, name):
    test_obj.checkEdgesLibraryUncut()

#### Jn Tests ####
# Check the Jn value vs the paper
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.allModules
@pytest.mark.paperCompare
def test_checkJn(test_obj, name):
    test_obj.checkJn()

# Check the Jn value vs the relative neighbourhood graph library
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.graphLibrary
@pytest.mark.allModules
def test_checkJn_lib(test_obj, name):
    test_obj.checkJn_lib()

# Check the Jn remains stable on dataset randomization
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.random
@pytest.mark.slow
@pytest.mark.allModules
def test_checkRandomize(test_obj, name):
    test_obj.checkRandomize()

#### N1 Tests ####
@pytest.mark.parametrize("test_obj, name", pytest.benchmarks)
@pytest.mark.benchmarks
@pytest.mark.N1
@pytest.mark.allModules
def test_checkN1(test_obj, name):
    test_obj.checkN1()