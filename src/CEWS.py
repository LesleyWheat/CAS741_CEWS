# Description

# Imports

# Project files
import inputModule
import calculationModule
import outputModule

# --------------------------------------------------------------------


def cutEdgeWeight(inData, inLabels):
    # Check input
    data, labels = inputModule.proccessInput(inData, inLabels)

    # Compute CEWS
    Jn = calculationModule.calc(data, labels)

    # Verify output
    outputModule.verifyOutput(Jn)

    return (Jn)
