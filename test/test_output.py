# Description

# Imports

# Core
import sys
import os

# Packages
import pytest

# Project files
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
import outputModule

# --------------------------------------------------------------------
# Tests


@pytest.mark.OM
def test_IM_good():
    JN = 0.5
    outputModule.verifyOutput(JN)


@pytest.mark.OM
@pytest.mark.VnV
def test_T14_IM_tooBig():
    JN = 1.1
    with pytest.raises(ValueError) as exc_info:
        outputModule.verifyOutput(JN)


@pytest.mark.OM
@pytest.mark.VnV
def test_T15_IM_tooSmall():
    JN = -0.1

    with pytest.raises(ValueError) as exc_info:
        outputModule.verifyOutput(JN)
