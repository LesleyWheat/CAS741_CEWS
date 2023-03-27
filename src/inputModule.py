# File description
# This module verifies the input data
#----------------------------------------------------------------
# Import libraries

# Packages
import numpy as np

#----------------------------------------------------------------
# Local variables
CLASS_MINIMUM = 2

#----------------------------------------------------------------
# Functions
def proccessInput(inData, inLabels):
    # Check types
    try:
        data = np.asarray(inData, dtype=np.float32)
    except Exception as e:
        raise TypeError("Data matrix is an invalid type or format")
    
    try: 
        labels = np.asarray(inLabels, dtype=np.int8)
    except Exception as e:
        raise TypeError("Label matrix is an invalid type or format")
    
    # check size
    if(data.shape[0] != labels.shape[0]):
        raise ValueError("Number of samples do not match number of labels")
    
    # check labels is n x 1
    if (labels.shape != (data.shape[0],)):
        raise TypeError("Labels need to be a vector")
    
    # Check number of classes
    classList = np.unique(labels)
    k = classList.shape[0]
    if(k < CLASS_MINIMUM):
        raise ValueError("Requires a minimum of two classes")

    
    return(data, labels)