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
import CEWS

#--------------------------------------------------------------------
# Functions to setup tests
def TC_good():
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 1, 2]
    return(data, labels)

def TC_moreLabels():
    # More labels than data
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 1, 2, 3]
    return(data, labels)

def TC_moreData():
    # More data than labels
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 1]
    return(data, labels)

def TC_sameLabels():
    # All same labels
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 0, 0]
    return(data, labels)

def TC_dataFormatSize():
    # Some observations have more features
    data = [[1, 0, 0],[0, 1],[0, 1]]
    labels = [0, 0, 1]
    return(data, labels)

def TC_dataFormatType():
    # Data has a string in it
    data = [[1, 0],[0, "?"],[0, 1]]
    labels = [0, 0, 1]
    return(data, labels)

def TC_labelFormatSize():
    # More labels have more values
    data = [[1, 0],[0, 1],[0, 1]]
    labels = [0, 0, [1, 1]]
    return(data, labels)

def TC_labelFormatType():
    # Label has a string
    data = [[1, 0],[0, 0],[0, 1]]
    labels = [0, "?", 1]
    return(data, labels)

def TC_labelNotVector():
    # Lablels is a matrix
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

# VnV Tests
@pytest.mark.IM
@pytest.mark.FP
def test_IM_good():
    inData, inLabels = TC_good()
    data, labels = CEWS.cutEdgeWeight(inData, inLabels)

@pytest.mark.FP
@pytest.mark.IM
def test_IM_wrongSizeLabel():
    # Too Many labels
    with pytest.raises(ValueError) as exc_info:
        inData, inLabels = TC_moreLabels()
        CEWS.cutEdgeWeight(inData, inLabels)

@pytest.mark.FP
@pytest.mark.IM
def test_IM_wrongSizeData():
    # Too much data
    with pytest.raises(ValueError) as exc_info:
        inData, inLabels = TC_moreData()
        CEWS.cutEdgeWeight(inData, inLabels)

@pytest.mark.FP
@pytest.mark.IM
def test_IM_oneClass():
    with pytest.raises(ValueError) as exc_info:
        inData, inLabels = TC_sameLabels()
        CEWS.cutEdgeWeight(inData, inLabels)

@pytest.mark.FP
@pytest.mark.IM
def test_IM_formatTypeData():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_dataFormatType()
        CEWS.cutEdgeWeight(inData, inLabels)

@pytest.mark.FP
@pytest.mark.IM
def test_IM_formatTypeLabel():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_labelFormatType()
        CEWS.cutEdgeWeight(inData, inLabels)

@pytest.mark.FP
@pytest.mark.IM
def test_IM_formatTypeData():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_dataFormatType()
        CEWS.cutEdgeWeight(inData, inLabels)

@pytest.mark.FP
@pytest.mark.IM
def test_IM_formatTypeLabel():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_labelFormatType()
        CEWS.cutEdgeWeight(inData, inLabels)

@pytest.mark.FP
@pytest.mark.IM
def test_IM_labelNotVector():
    with pytest.raises(TypeError) as exc_info:
        inData, inLabels = TC_labelNotVector()
        CEWS.cutEdgeWeight(inData, inLabels)