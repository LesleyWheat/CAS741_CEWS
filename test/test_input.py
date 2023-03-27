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
import inputModule

#--------------------------------------------------------------------
# Functions to setup tests
def TC_good():
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 1, 2]
    return(data, labels)

def TC_moreLabels():
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 1, 2, 3]
    return(data, labels)

def TC_moreData():
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 1]
    return(data, labels)

def TC_sameLabels():
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 0, 0]
    return(data, labels)

def TC_dataFormatSize():
    data = [[1, 0, 0],[0, 1],[0, 1]]
    labels = [0, 0, 1]
    return(data, labels)

def TC_dataFormatType():
    data = [[1, 0],[0, "?"],[0, 1]]
    labels = [0, 0, 1]
    return(data, labels)

def TC_labelFormatSize():
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 0, [1, 1]]
    return(data, labels)

def TC_labelFormatType():
    data = [[1, 0],[0, 0],[0, 1]]
    labels = [0, "?", 1]
    return(data, labels)

def TC_labelNotVector():
    data = [[1, 0],[0, 0],[0, 1]]
    labels = [[1, 0],[0, 0],[0, 1]]
    return(data, labels)

#--------------------------------------------------------------------
# Tests

@pytest.mark.IM
def test_IM_good():
    inData, inLabels = TC_good()
    data, labels = inputModule.proccessInput(inData, inLabels)

@pytest.mark.IM
def test_IM_wrongSizeLabel():
    # Too Many labels
    with pytest.raises(ValueError) as exc_info:
        inData, inLabels = TC_moreLabels()
        inputModule.proccessInput(inData, inLabels)
    
@pytest.mark.IM
def test_IM_wrongSizeData():
    # Too much data
    with pytest.raises(ValueError) as exc_info:
        inData, inLabels = TC_moreData()
        inputModule.proccessInput(inData, inLabels)

@pytest.mark.IM
def test_IM_oneClass():
    with pytest.raises(ValueError) as exc_info:
        inData, inLabels = TC_sameLabels()
        inputModule.proccessInput(inData, inLabels)

@pytest.mark.IM
def test_IM_formatTypeData():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_dataFormatType()
        inputModule.proccessInput(inData, inLabels)


@pytest.mark.IM
def test_IM_formatTypeLabel():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_labelFormatType()
        inputModule.proccessInput(inData, inLabels)


@pytest.mark.IM
def test_IM_formatTypeData():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_dataFormatType()
        inputModule.proccessInput(inData, inLabels)
    
@pytest.mark.IM
def test_IM_formatTypeLabel():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_labelFormatType()
        inputModule.proccessInput(inData, inLabels)

@pytest.mark.IM
def test_IM_labelNotVector():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_labelNotVector()
        inputModule.proccessInput(inData, inLabels)