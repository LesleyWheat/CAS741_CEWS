# File description
# This modules verifies the output value
#----------------------------------------------------------------
# Local variables

# Constants
JN_LOWERBOUND = 0
JN_UPPERBOUND = 1

#----------------------------------------------------------------
# Functions

def verifyOutput(Jn):
    if (Jn > JN_UPPERBOUND or Jn < JN_LOWERBOUND):
        raise ValueError("CEWS bound exceeded. Program error.")